"""
Stage-3 CoLLM: direct triplet training from base CLIP and SFR models.

This keeps the paper's CIR-triplet formulation:

    c_i = p(Phi([g(f(v_ref)); w_i]))
    z_i = f(v_target)
    L   = L_cl(c_i, z_i)

Unlike the paper's Stage-2 fine-tuning, this model does not require a Stage-1
checkpoint. It starts from the base CLIP and SFR models and trains:

  * optional LoRA on the CLIP vision transformer
  * LoRA on SFR-Embedding-2_R
  * the full image adapter
  * the full output projection
  * the contrastive logit scale

The CLIP and SFR base weights remain frozen.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModel, BitsAndBytesConfig, CLIPModel


def resolve_attn_implementation(requested: str = "auto") -> str:
    if requested == "auto":
        return "sdpa"
    if requested in ("sdpa", "eager"):
        return requested
    raise ValueError(f"Unsupported attention implementation: {requested}")


def resolve_llm_quantization(requested: str = "auto") -> bool:
    if requested == "4bit":
        return True
    if requested == "bf16":
        return False
    try:
        import bitsandbytes  # noqa: F401

        return True
    except ImportError:
        print("WARNING: bitsandbytes is unavailable; loading SFR in bf16/fp16.")
        return False


def resolve_compute_dtype(device=None) -> torch.dtype:
    if device is not None and device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= 8 else torch.float16
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


class ImageAdapter(nn.Module):
    def __init__(self, clip_dim: int = 1024, llm_dim: int = 4096):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(clip_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, features):
        return self.proj(features).unsqueeze(1)


class ProjectionHead(nn.Module):
    def __init__(self, llm_dim: int = 4096, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Linear(llm_dim, embed_dim)

    def forward(self, features):
        features = features.to(self.proj.weight.dtype)
        return F.normalize(self.proj(features), dim=-1)


def last_token_pool(last_hidden_states, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


class CoLLMStage3(nn.Module):
    def __init__(
        self,
        tokenizer,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        llm_model_name: str = "Salesforce/SFR-Embedding-2_R",
        clip_dim: int = 1024,
        llm_dim: int = 4096,
        embed_dim: int = 768,
        clip_lora_rank: int = 16,
        llm_lora_rank: int = 16,
        clip_lora_dropout: float = 0.1,
        llm_lora_dropout: float = 0.1,
        llm_precision: str = "auto",
        attn_implementation: str = "auto",
        gradient_checkpointing: bool = False,
        freeze_clip_lora: bool = False,
        device=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.use_4bit = resolve_llm_quantization(llm_precision)
        self.compute_dtype = resolve_compute_dtype(device)
        self.gradient_checkpointing = gradient_checkpointing
        self.clip_lora_frozen = freeze_clip_lora

        # CLIP base stays frozen; LoRA is attached only to the vision tower.
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        for parameter in self.clip.parameters():
            parameter.requires_grad = False

        clip_lora = LoraConfig(
            r=clip_lora_rank,
            lora_alpha=clip_lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=clip_lora_dropout,
            bias="none",
        )
        self.clip.vision_model = get_peft_model(self.clip.vision_model, clip_lora)

        # The fixed CLIP visual projection defines the target retrieval space.
        for parameter in self.clip.visual_projection.parameters():
            parameter.requires_grad = False
        self.set_clip_lora_trainable(not freeze_clip_lora)

        attn_impl = resolve_attn_implementation(attn_implementation)
        llm_kwargs = {
            "attn_implementation": attn_impl,
            "torch_dtype": self.compute_dtype,
        }
        if self.use_4bit:
            llm_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            if device is not None and device.type == "cuda":
                llm_kwargs["device_map"] = {"": device.index or 0}

        self.llm = AutoModel.from_pretrained(llm_model_name, **llm_kwargs)
        if self.use_4bit:
            self.llm = prepare_model_for_kbit_training(
                self.llm,
                use_gradient_checkpointing=gradient_checkpointing,
            )
        if gradient_checkpointing:
            self.llm.gradient_checkpointing_enable()

        self.image_token = "<image>"
        if self.image_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([self.image_token])
        current_vocab_size = self.llm.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) != current_vocab_size:
            self.llm.resize_token_embeddings(len(self.tokenizer))
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

        llm_lora = LoraConfig(
            r=llm_lora_rank,
            lora_alpha=llm_lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=llm_lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.llm = get_peft_model(self.llm, llm_lora)

        self.image_adapter = ImageAdapter(clip_dim, llm_dim).to(torch.float32)
        self.projection = ProjectionHead(llm_dim, embed_dim).to(torch.float32)
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / 0.07), dtype=torch.float32)
        )

        if device is not None:
            self.place_modules(device)

    def place_modules(self, device):
        self.clip.to(device)
        self.image_adapter.to(device)
        self.projection.to(device)
        self.logit_scale.data = self.logit_scale.data.to(device)
        if not self.use_4bit:
            self.llm.to(device)

    def set_clip_lora_trainable(self, trainable: bool):
        for name, parameter in self.clip.vision_model.named_parameters():
            if "lora_" in name:
                parameter.requires_grad = trainable
        self.clip_lora_frozen = not trainable

    @property
    def llm_device(self):
        return next(self.llm.parameters()).device

    def _build_instruction(self, text: str) -> str:
        return (
            "Instruct: Find the image that matches the query.\n"
            f"Query:\nImage: {self.image_token}\nText: {text}"
        )

    def encode_reference(self, pixel_values):
        outputs = self.clip.vision_model(pixel_values=pixel_values)
        return F.normalize(outputs.pooler_output, dim=-1)

    def encode_target(self, pixel_values):
        outputs = self.clip.vision_model(pixel_values=pixel_values)
        image_embeds = self.clip.visual_projection(outputs.pooler_output)
        return F.normalize(image_embeds, dim=-1)

    def encode_query(self, visual_token, texts, device=None):
        if device is None:
            device = self.llm_device

        instructions = [self._build_instruction(text) for text in texts]
        encoded = self.tokenizer(
            instructions,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)

        inputs_embeds = self.llm.get_input_embeddings()(encoded["input_ids"])
        image_mask = encoded["input_ids"] == self.image_token_id
        image_token_counts = image_mask.sum(dim=1)
        if not torch.all(image_token_counts == 1):
            raise RuntimeError(
                "Every instruction must contain exactly one <image> token; "
                f"got counts {image_token_counts.tolist()}"
            )

        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[image_mask] = visual_token.squeeze(1).to(inputs_embeds.dtype)
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=encoded["attention_mask"],
        )
        pooled = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
        return self.projection(pooled)

    def forward_triplet_microbatched(
        self,
        ref_imgs,
        target_imgs,
        mod_texts,
        micro_batch: int,
        device=None,
    ):
        if device is None:
            device = self.llm_device

        # These CLIP forwards retain gradients for the vision LoRA.
        reference_features = self.encode_reference(ref_imgs)
        target_features = self.encode_target(target_imgs)
        adapter_dtype = next(self.image_adapter.parameters()).dtype
        visual_tokens = self.image_adapter(reference_features.to(adapter_dtype))

        query_parts = []
        for start in range(0, ref_imgs.shape[0], micro_batch):
            end = min(start + micro_batch, ref_imgs.shape[0])
            query_parts.append(
                self.encode_query(
                    visual_token=visual_tokens[start:end].to(device),
                    texts=mod_texts[start:end],
                    device=device,
                )
            )

        return torch.cat(query_parts, dim=0), target_features


def build_stage3_model(
    tokenizer,
    clip_model_name="openai/clip-vit-large-patch14",
    llm_model_name="Salesforce/SFR-Embedding-2_R",
    clip_dim=1024,
    llm_dim=4096,
    embed_dim=768,
    clip_lora_rank=16,
    llm_lora_rank=16,
    llm_precision="auto",
    attn_implementation="eager",
    gradient_checkpointing=False,
    freeze_clip_lora=False,
    device=None,
):
    return CoLLMStage3(
        tokenizer=tokenizer,
        clip_model_name=clip_model_name,
        llm_model_name=llm_model_name,
        clip_dim=clip_dim,
        llm_dim=llm_dim,
        embed_dim=embed_dim,
        clip_lora_rank=clip_lora_rank,
        llm_lora_rank=llm_lora_rank,
        llm_precision=llm_precision,
        attn_implementation=attn_implementation,
        gradient_checkpointing=gradient_checkpointing,
        freeze_clip_lora=freeze_clip_lora,
        device=device,
    )
