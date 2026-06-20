"""
model_stage2.py — CoLLM Stage-2 (CIR-triplet fine-tuning) model definition.

Implements the triplet branch of CoLLM (paper Sec. 3.3, implementation
details in Sec. 9.2):
  - vision encoder f(.) is FROZEN (no LoRA at all) — "vision encoder
    features are already aligned during the pre-training phase" (Sec 9.2).
  - only the LLM Phi(.) is fine-tuned, with a *fresh*, smaller LoRA adapter
    (rank = alpha = 16, vs. rank 64 used in pre-training; Sec 9.2).
  - image_adapter g(.) and projection p(.) are carried over from Stage-1
    and remain trainable — they are not "the vision encoder", and the
    paper gives no indication they should be frozen.
  - no Slerp / nearest-neighbor / modification-text synthesis is needed —
    Stage-2 trains on REAL (reference, target, modification_text) triplets.
  - training objective is a single contrastive loss L = L_cl(c_i, z_i)
    (Sec. 3.3), not the 3-way average used in Stage 1.

Run merge_stage1_checkpoint.py first — this file assumes the Stage-1 LoRA
adapters have already been baked into clean base weights. See that script's
docstring for why the merge step is required before re-LoRA-ing at rank 16.

The small utility functions/classes below are duplicated from your
Stage-1 model.py (with one fix — see resolve_bnb_compute_dtype) so this
file is standalone. If model.py is importable in your package, feel free
to delete the duplicates and import from there instead, just keep the
resolve_bnb_compute_dtype fix.
"""

from pathlib import Path

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
        print("WARNING: bitsandbytes not installed — loading LLM in bf16.")
        return False


def resolve_compute_dtype(device=None) -> torch.dtype:
    if device is not None and device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= 8 else torch.float16
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def resolve_bnb_compute_dtype(device=None) -> torch.dtype:
    """
    FIX vs. your Stage-1 model.py: that version hardcodes bf16 here
    regardless of GPU. On Turing GPUs (e.g. Quadro RTX 8000, sm_75) there
    is no native bf16 tensor-core support, so 4-bit matmuls in bf16 either
    run through a slow fallback path or, depending on your bitsandbytes
    version, error outright. This picks fp16 on pre-Ampere GPUs instead —
    apply the same fix in model.py if you re-run Stage-1 on this machine.
    """
    return resolve_compute_dtype(device)


class ImageAdapter(nn.Module):
    """Must match Stage-1's ImageAdapter exactly — we load its weights."""

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
    """Must match Stage-1's ProjectionHead exactly — we load its weights."""

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


class CoLLMStage2(nn.Module):
    """
    Stage-2 (triplet fine-tuning) model.

    Differences from CoLLMStage1 (model.py):
      * self.clip is a PLAIN CLIPModel (no PEFT wrapper) — entirely frozen.
      * self.llm carries the Stage-1 LoRA *merged* into its base weights
        (via merge_stage1_checkpoint.py), then wrapped with a fresh,
        smaller LoRA adapter (default r=alpha=16).
      * No reference-embedding / modification-text synthesis modules —
        call encode_reference() directly on the real reference image.
    """

    def __init__(
        self,
        tokenizer,
        merged_clip_dir: str,
        merged_llm_dir: str,
        llm_dim: int = 4096,
        clip_dim: int = 1024,
        embed_dim: int = 768,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        llm_precision: str = "auto",
        attn_implementation: str = "auto",
        gradient_checkpointing: bool = False,
        device=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.use_4bit = resolve_llm_quantization(llm_precision)
        self.llm_dim = llm_dim
        self.bnb_compute_dtype = resolve_bnb_compute_dtype(device) if self.use_4bit else None
        self.compute_dtype = self.bnb_compute_dtype if self.use_4bit else resolve_compute_dtype(device)
        self.gradient_checkpointing = gradient_checkpointing

        # ---- Vision encoder: frozen, no LoRA (paper Sec. 9.2) -------------
        self.clip = CLIPModel.from_pretrained(merged_clip_dir)
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.eval()

        # ---- LLM: load Stage-1-merged weights, then add a NEW small LoRA -
        attn_impl = resolve_attn_implementation(attn_implementation)
        llm_kwargs = {"attn_implementation": attn_impl, "torch_dtype": self.compute_dtype}
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

        self.llm = AutoModel.from_pretrained(merged_llm_dir, **llm_kwargs)
        if self.use_4bit:
            self.llm = prepare_model_for_kbit_training(
                self.llm, use_gradient_checkpointing=self.gradient_checkpointing
            )
        if self.gradient_checkpointing:
            self.llm.gradient_checkpointing_enable()

        # Same image-token bookkeeping as Stage-1 (CoLLMStage1.__init__).
        # The embedding row's *value* is irrelevant — it is always
        # overwritten by the projected visual token in encode_query()
        # below — only its existence/index matters, so redoing this here
        # against the merged (unresized) checkpoint is safe.

        #Changed
        self.image_token = "<image>"

        if self.image_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([self.image_token])

        current_vocab_size = self.llm.get_input_embeddings().weight.shape[0]

        if len(self.tokenizer) != current_vocab_size:
            self.llm.resize_token_embeddings(len(self.tokenizer))

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(
            self.image_token
        )

        llm_lora = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.llm = get_peft_model(self.llm, llm_lora)

        # ---- Bridging modules: carried over from Stage-1, stay trainable -
        self.image_adapter = ImageAdapter(clip_dim, llm_dim).to(torch.float32)
        self.projection = ProjectionHead(llm_dim, embed_dim).to(torch.float32)
        self.logit_scale = nn.Parameter(torch.zeros([]))  # overwritten by load_stage1_bridge_weights()

        if device is not None:
            self.place_trainable_modules(device)

    def place_trainable_modules(self, device):
        self.clip.to(device)
        self.image_adapter.to(device)
        self.projection.to(device)
        self.logit_scale.data = self.logit_scale.data.to(device)
        if not self.use_4bit:
            self.llm.to(device)

    @property
    def llm_device(self):
        return next(self.llm.parameters()).device
    
    def train(self, mode: bool = True):
        """
        Keep CLIP frozen even when model.train() is called.
        """

        super().train(mode)

        self.clip.eval()

        return self

    def load_stage1_bridge_weights(self, extra_modules_pt: str, device):
        """Load image_adapter / projection / logit_scale from the Stage-1
        checkpoint's extra_modules.pt (carried through unchanged by
        merge_stage1_checkpoint.py — they're never touched by the LoRA
        merge, since they're not part of clip or llm)."""
        extra = torch.load(
            extra_modules_pt,
            map_location="cpu",
        )
        self.image_adapter.load_state_dict(extra["image_adapter"])
        self.projection.load_state_dict(extra["projection"])
        self.logit_scale.data = (
            extra["logit_scale"]
            .float()
            .to(device)
        )
    @torch.no_grad()
    def encode_target(self, pixel_values):
        outputs = self.clip.vision_model(pixel_values=pixel_values)
        image_embeds = self.clip.visual_projection(outputs.pooler_output)
        return F.normalize(image_embeds, dim=-1)

    @torch.no_grad()
    def encode_reference(self, pixel_values):
        outputs = self.clip.vision_model(pixel_values=pixel_values)
        return F.normalize(outputs.pooler_output, dim=-1)

    def _build_instruction(self, has_image: bool, text=None) -> str:
        prompt = "Instruct: Find the image that matches the query.\nQuery:\n"
        if has_image:
            prompt += f"Image: {self.image_token}\n"
        if text is not None:
            prompt += f"Text: {text}"
        return prompt.strip()

    def encode_query(self, visual_token, texts, device=None):
        """Composed query c_i = p(Phi([g(h_i); w_i])) — paper Sec. 3.3."""
        if device is None:
            device = self.llm_device
        batch_size = visual_token.shape[0]
        instructions = [self._build_instruction(True, texts[i]) for i in range(batch_size)]
        encoded = self.tokenizer(
            instructions, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(device)
        inputs_embeds = self.llm.get_input_embeddings()(encoded["input_ids"])
        mask = encoded["input_ids"] == self.image_token_id
        vt = visual_token.squeeze(1).to(inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[mask] = vt
        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=encoded["attention_mask"])
        pooled = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
        return self.projection(pooled)

    def forward_triplet(self, ref_imgs, target_imgs, mod_texts, device=None):
        """One forward pass for a batch of REAL CIR triplets. Returns
        (c, z) — the composed query embedding and target embedding used in
        the single contrastive loss L_cl(c_i, z_i)."""
        if device is None:
            device = self.llm_device
        h = self.encode_reference(ref_imgs)             # frozen CLIP, [B, 1024]
        z = self.encode_target(target_imgs)              # frozen CLIP, [B, 768]
        adapter_dtype = next(self.image_adapter.parameters()).dtype
        visual_token = self.image_adapter(h.to(adapter_dtype))
        visual_token = visual_token.to(device)
        c = self.encode_query(visual_token=visual_token, texts=mod_texts, device=device)
        return c, z

    def forward_triplet_microbatched(self, ref_imgs, target_imgs, mod_texts, micro_batch, device=None):
        """Micro-batches only the LLM forward (the memory bottleneck),
        mirroring forward_llm_queries_microbatched() in Stage-1's model.py
        — but for the single triplet query instead of three."""
        if device is None:
            device = self.llm_device
        h = self.encode_reference(ref_imgs)
        z = self.encode_target(target_imgs)
        adapter_dtype = next(self.image_adapter.parameters()).dtype
        visual_token = self.image_adapter(h.to(adapter_dtype))
        n = ref_imgs.shape[0]
        c_parts = []
        for start in range(0, n, micro_batch):
            end = min(start + micro_batch, n)
            vt = visual_token[start:end].to(device)
            texts = mod_texts[start:end]
            c_parts.append(self.encode_query(visual_token=vt, texts=texts, device=device))
        c = torch.cat(c_parts, dim=0)
        return c, z


def build_stage2_model(
    tokenizer,
    merged_dir: str,
    llm_dim: int = 4096,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    llm_precision: str = "auto",
    attn_implementation: str = "eager",
    gradient_checkpointing: bool = False,
    device=None,
) -> CoLLMStage2:
    """Convenience constructor matching the directory layout written by
    merge_stage1_checkpoint.py:
        merged_dir/clip_merged/        — plain CLIPModel weights
        merged_dir/llm_merged/         — plain AutoModel weights (Stage-1
                                          LoRA already merged in)
        merged_dir/extra_modules.pt    — image_adapter / projection /
                                          logit_scale carried over as-is
    """
    merged_dir = Path(merged_dir)
    model = CoLLMStage2(
        tokenizer=tokenizer,
        merged_clip_dir=str(merged_dir / "clip_merged"),
        merged_llm_dir=str(merged_dir / "llm_merged"),
        llm_dim=llm_dim,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        llm_precision=llm_precision,
        attn_implementation=attn_implementation,
        gradient_checkpointing=gradient_checkpointing,
        device=device,
    )
    model.load_stage1_bridge_weights(str(merged_dir / "extra_modules.pt"), device)
    return model