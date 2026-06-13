import torch
import torch.nn.functional as F

def slerp(v0, v1, t=0.5):
    """
    Spherical Linear Interpolation.
    Paper equation 2: h*_i = sin(α·θ)/sin(θ) · h'_i + sin((1-α)·θ)/sin(θ) · h'_j
    
    Args:
        v0: h'_i  shape [batch, dim]  (target image augmented embedding)
        v1: h'_j  shape [batch, dim]  (nearest neighbor embedding)
        t:  α in paper, default 0.5 (optimal per Figure 7a ablation)
    
    Returns:
        h*_i  shape [batch, dim]  (synthesized reference embedding)
    """
    # Normalize inputs — CLIP outputs are already normalized
    # but defensive normalization prevents numerical issues
    v0 = F.normalize(v0, dim=-1)
    v1 = F.normalize(v1, dim=-1)
    
    # cos(θ) = v0 · v1  (valid since both unit vectors)
    dot = (v0 * v1).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    # clamp to [-1, 1] prevents acos domain errors from floating point
    
    # Fallback: if vectors nearly parallel, use normalized linear interp
    # avoids sin(θ) ≈ 0 in denominator
    close_condition = (torch.abs(dot) > 0.9995)
    linear_interp = F.normalize(v0 + t * (v1 - v0), dim=-1)
    
    # Slerp
    theta = torch.acos(dot)           # shape [batch, 1]
    sin_theta = torch.sin(theta)      # shape [batch, 1]
    
    # Paper: sin(α·θ)/sin(θ) · h'_i + sin((1-α)·θ)/sin(θ) · h'_j
    scale_i = torch.sin(t * theta) / sin_theta        # α=t for v0=h'_i
    scale_j = torch.sin((1.0 - t) * theta) / sin_theta # (1-α) for v1=h'_j
    
    slerp_interp = scale_i * v0 + scale_j * v1
    slerp_interp = F.normalize(slerp_interp, dim=-1)  # safety normalization
    
    return torch.where(close_condition, linear_interp, slerp_interp)