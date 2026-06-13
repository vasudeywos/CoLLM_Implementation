from transformers import CLIPModel, CLIPProcessor
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageAdapter(nn.Module):
    """
    Adapter g(·) from paper Sec 3.1.
    Maps single global CLIP embedding → single LLM token.
    
    Input:  [B, clip_proj_dim]  — projected, normalized CLIP embedding
                                  768 for CLIP-L/14
                                  512 for CLIP-B/32
    Output: [B, 1, llm_dim]    — single visual token in LLM embedding space
    
    Architecture: mlp2x_gelu following LLaVA 1.5 style (Sec 9.1 ref to LLaVA)
    but applied to global CLS vector, NOT patch sequence.
    """
    def __init__(self, clip_proj_dim: int = 768, llm_dim: int = 4096):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(clip_proj_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )
    
    def forward(self, h_star: torch.Tensor) -> torch.Tensor:
        # h_star: [B, clip_proj_dim] — output of slerp, already unit norm
        x = self.proj(h_star)      # [B, llm_dim]
        return x.unsqueeze(1)      # [B, 1, llm_dim] — single visual token


def slerp(h_i: torch.Tensor, h_j: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """
    Spherical Linear Interpolation — Eq. (2) from paper.
    Inputs must be L2-normalized (unit vectors).
    Output is also unit norm (property of Slerp on unit sphere).
    """
    # Safety clamp to avoid acos domain errors at exactly ±1
    dot = (h_i * h_j).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(dot)                              # [B, 1]
    sin_theta = torch.sin(theta)                         # [B, 1]
    
    coeff_i = torch.sin(alpha * theta) / sin_theta       # [B, 1]
    coeff_j = torch.sin((1 - alpha) * theta) / sin_theta # [B, 1]
    
    return coeff_i * h_i + coeff_j * h_j                # [B, clip_proj_dim]


def get_normalized_clip_embeddings(
    clip_model: CLIPModel,
    pixel_values: torch.Tensor
) -> torch.Tensor:
    """
    Returns L2-normalized projected CLIP image embeddings.
    Uses get_image_features() which applies the final projection layer.
    
    For CLIP-L/14: output dim = 768
    For CLIP-B/32: output dim = 512
    """
    # get_image_features() = encode + project (no normalization applied by HF)
    embeds = clip_model.get_image_features(pixel_values=pixel_values)  # [B, 768]
    return F.normalize(embeds, p=2, dim=-1)                             # [B, 768]


def stage1_forward(
    clip_model,
    adapter,
    llm,
    proj_head,
    pixel_values,        # [B, 3, H, W] — augmented images v'_i
    pixel_values_target, # [B, 3, H, W] — original images v_i (for z_i)
    w_i_input_ids,       # [B, T] — tokenized target captions w_i
    w_star_input_ids,    # [B, T'] — tokenized synthetic mod texts w*_i
    alpha: float = 0.5
):
    B = pixel_values.shape[0]

    # ── 1. Encode augmented batch → normalized embeddings ────────────────────
    h_prime = get_normalized_clip_embeddings(clip_model, pixel_values)
    # h_prime: [B, 768], unit norm

    # ── 2. In-batch nearest neighbor — Eq. (1) ──────────────────────────────
    sim_matrix = h_prime @ h_prime.T              # [B, B]
    sim_matrix.fill_diagonal_(-float('inf'))      # exclude self
    j_indices = sim_matrix.argmax(dim=1)          # [B]
    h_prime_j = h_prime[j_indices]               # [B, 768]

    # ── 3. Slerp — Eq. (2) ──────────────────────────────────────────────────
    h_star = slerp(h_prime, h_prime_j, alpha=alpha)
    # h_star: [B, 768], unit norm (Slerp preserves this)

    # ── 4. Adapter g(·) → single visual token ───────────────────────────────
    visual_token = adapter(h_star)
    # visual_token: [B, 1, 4096]

    # ── 5. Target embedding z_i = f(v_i) ────────────────────────────────────
    # Uses ORIGINAL (non-augmented) images
    z_i = get_normalized_clip_embeddings(clip_model, pixel_values_target)
    # z_i: [B, 768]

    # ── 6. Three branches through shared LLM Φ(·) ───────────────────────────

    # Branch 1: c^v — image-only, Eq. (3)
    c_v = llm_forward_with_embeds(llm, proj_head, inputs_embeds=visual_token)

    # Branch 2: c^w — text-only using target caption w_i, Eq. (4)
    c_w = llm_forward_with_ids(llm, proj_head,
                                input_ids=w_i_input_ids)

    # Branch 3: c_i — composed, Eq. (5)
    # Prepend visual token to text embeddings
    text_embeds = llm.model.embed_tokens(w_star_input_ids)   # [B, T', 4096]
    combined = torch.cat([visual_token, text_embeds], dim=1) # [B, 1+T', 4096]
    text_mask = (w_star_input_ids != tokenizer.pad_token_id).long()
    visual_mask = torch.ones(B, 1, device=pixel_values.device)
    combined_mask = torch.cat([visual_mask, text_mask], dim=1)
    c_i = llm_forward_with_embeds(llm, proj_head,
                                   inputs_embeds=combined,
                                   attention_mask=combined_mask)

    return c_v, c_w, c_i, z_i


def llm_forward_with_embeds(llm, proj_head, inputs_embeds, attention_mask=None):
    """Last token hidden state → proj → retrieval embedding."""
    out = llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    last = out.last_hidden_state[:, -1, :]   # [B, 4096]
    return proj_head(last)                   # [B, out_dim]


def llm_forward_with_ids(llm, proj_head, input_ids, attention_mask=None):
    out = llm(input_ids=input_ids, attention_mask=attention_mask)
    last = out.last_hidden_state[:, -1, :]
    return proj_head(last)