import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, BitsAndBytesConfig, CLIPModel
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training


def resolve_attn_implementation(requested: str = "auto") -> str:
    if requested == "auto":
        return "sdpa"
    if requested in ("sdpa", "eager"):
        return requested
    raise ValueError(f"Unsupported attention implementation: {requested}")


def resolve_llm_quantization(requested: str = "auto") -> bool:
    """Pick 4-bit QLoRA by default when bitsandbytes is available."""
    if requested == "4bit":
        return True
    if requested == "bf16":
        return False
    try:
        import bitsandbytes  # noqa: F401
        return True
    except ImportError:
        print(
            "WARNING: bitsandbytes not installed — loading LLM in bf16. "
            "This will likely OOM on a 48GB GPU. "
            "Install bitsandbytes to enable automatic 4-bit QLoRA."
        )
        return False


def resolve_compute_dtype(device=None):
    if device is not None and device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= 8 else torch.float16
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def resolve_bnb_compute_dtype() -> torch.dtype:
    """4-bit QLoRA matmuls always use bf16 for numerical stability (incl. Turing GPUs)."""
    return torch.bfloat16


class ImageAdapter(nn.Module):
    def __init__(self, clip_dim: int = 1024, llm_dim: int = 4096):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(clip_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, x):
        return self.proj(x).unsqueeze(1)


class ProjectionHead(nn.Module):
    def __init__(self, llm_dim: int = 4096, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Linear(llm_dim, embed_dim)

    def forward(self, x):
        x = x.to(self.proj.weight.dtype)
        return F.normalize(self.proj(x), dim=-1)


def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


class CoLLMStage1(nn.Module):
    def __init__(
        self,
        tokenizer,
        clip_model_name="openai/clip-vit-large-patch14",
        llm_model_name="Salesforce/SFR-Embedding-2_R",
        lora_rank=32,
        clip_dim=1024,
        llm_dim=4096,
        embed_dim=768,
        llm_precision="auto",
        attn_implementation="auto",
        gradient_checkpointing=False,
        device=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.use_4bit = resolve_llm_quantization(llm_precision)
        self.llm_dim = llm_dim
        self.bnb_compute_dtype = resolve_bnb_compute_dtype() if self.use_4bit else None
        self.compute_dtype = (
            self.bnb_compute_dtype if self.use_4bit else resolve_compute_dtype(device)
        )
        self.gradient_checkpointing = gradient_checkpointing

        self.clip = CLIPModel.from_pretrained(clip_model_name)
        clip_lora = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=0.1,
            bias="none",
        )
        self.clip.vision_model = get_peft_model(self.clip.vision_model, clip_lora)
        for p in self.clip.text_model.parameters():
            p.requires_grad = False
        for p in self.clip.text_projection.parameters():
            p.requires_grad = False

        attn_impl = resolve_attn_implementation(attn_implementation)
        llm_kwargs = {
            "attn_implementation": attn_impl,
            "torch_dtype": self.compute_dtype,
        }

        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.bnb_compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            llm_kwargs["quantization_config"] = bnb_config
            if device is not None and device.type == "cuda":
                llm_kwargs["device_map"] = {"": device.index or 0}

        self.llm = AutoModel.from_pretrained(llm_model_name, **llm_kwargs)
        if self.use_4bit:
            self.llm = prepare_model_for_kbit_training(self.llm)
        if self.gradient_checkpointing:
            self.llm.gradient_checkpointing_enable()

        self.image_token = "<image>"
        if self.image_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([self.image_token])
            self.llm.resize_token_embeddings(len(self.tokenizer))
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

        llm_lora = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.llm = get_peft_model(self.llm, llm_lora)

        self.image_adapter = ImageAdapter(clip_dim, llm_dim).to(torch.float32)
        self.projection = ProjectionHead(llm_dim, embed_dim).to(torch.float32)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if device is not None:
            self.place_trainable_modules(device)

    def place_trainable_modules(self, device):
        """Move non-quantized modules; 4-bit LLM stays on its device_map target."""
        self.clip.to(device)
        self.image_adapter.to(device)
        self.projection.to(device)
        self.logit_scale.data = self.logit_scale.data.to(device)
        if not self.use_4bit:
            self.llm.to(device)

    @property
    def llm_device(self):
        return next(self.llm.parameters()).device

    def encode_target(self, pixel_values):
        outputs = self.clip.vision_model(pixel_values=pixel_values)
        image_embeds = self.clip.visual_projection(outputs.pooler_output)
        return F.normalize(image_embeds, dim=-1)

    def encode_reference(self, pixel_values):
        outputs = self.clip.vision_model(pixel_values=pixel_values)
        return F.normalize(outputs.pooler_output, dim=-1)

    def _build_instruction(self, has_image: bool, text: str | None = None) -> str:
        prompt = "Instruct: Find the image that matches the query.\nQuery:\n"
        if has_image:
            prompt += f"Image: {self.image_token}\n"
        if text is not None:
            prompt += f"Text: {text}"
        return prompt.strip()

    def encode_query(self, visual_token=None, texts=None, device=None):
        if device is None:
            device = self.llm_device

        if visual_token is not None:
            batch_size = visual_token.shape[0]
        else:
            batch_size = len(texts)

        instructions = []
        for i in range(batch_size):
            has_image = visual_token is not None
            text = texts[i] if texts is not None else None
            instructions.append(self._build_instruction(has_image, text))

        encoded = self.tokenizer(
            instructions,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)

        inputs_embeds = self.llm.get_input_embeddings()(encoded["input_ids"])

        if visual_token is not None:
            mask = encoded["input_ids"] == self.image_token_id
            vt = visual_token.squeeze(1).to(inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[mask] = vt

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=encoded["attention_mask"],
        )
        pooled = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
        return self.projection(pooled)

    def forward_queries(self, h_star, captions, mod_texts, device=None):
        if device is None:
            device = self.llm_device

        adapter_dtype = next(self.image_adapter.parameters()).dtype
        visual_token = self.image_adapter(h_star.to(adapter_dtype))

        c_v = self.encode_query(visual_token=visual_token, texts=None, device=device)
        c_w = self.encode_query(visual_token=None, texts=captions, device=device)
        c = self.encode_query(visual_token=visual_token, texts=mod_texts, device=device)

        return c_v, c_w, c

    def forward_llm_queries_microbatched(
        self,
        h_star,
        captions,
        mod_texts,
        micro_batch: int,
        device=None,
    ):
        """
        Micro-batch only the three LLM encode_query passes.

        h_star must already come from a full-batch CLIP forward (no CLIP chunking).
        The image adapter runs once on the full batch; only Phi(.) is chunked.
        """
        if device is None:
            device = self.llm_device

        adapter_dtype = next(self.image_adapter.parameters()).dtype
        visual_token = self.image_adapter(h_star.to(adapter_dtype))
        n = h_star.shape[0]
        c_v_parts, c_w_parts, c_parts = [], [], []

        for start in range(0, n, micro_batch):
            end = min(start + micro_batch, n)
            vt = visual_token[start:end]
            caps = captions[start:end]
            mods = mod_texts[start:end]

            c_v_parts.append(self.encode_query(visual_token=vt, texts=None, device=device))
            c_w_parts.append(self.encode_query(visual_token=None, texts=caps, device=device))
            c_parts.append(self.encode_query(visual_token=vt, texts=mods, device=device))

            # if device.type == "cuda":
            #     torch.cuda.empty_cache()

        return torch.cat(c_v_parts, dim=0), torch.cat(c_w_parts, dim=0), torch.cat(c_parts, dim=0)
