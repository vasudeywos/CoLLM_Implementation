import torch
import torch.nn.functional as F


def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    """
    Spherical Linear Interpolation.
    
    Paper Equation 2:
        theta    = arccos(h'_i · h'_j)
        h*_i     = sin(alpha*theta)/sin(theta) * h'_i
                 + sin((1-alpha)*theta)/sin(theta) * h'_j
    
    Args:
        v0: h'_i  shape [B, D]  target image augmented embedding
        v1: h'_j  shape [B, D]  nearest neighbor embedding  
        t:  alpha in paper, default 0.5 per Figure 7a ablation
    
    Returns:
        h*_i  shape [B, D]  synthesized reference embedding
    """
    # Both must be unit vectors for arccos to be valid
    v0 = F.normalize(v0, dim=-1)
    v1 = F.normalize(v1, dim=-1)

    # cos(theta) = v0 · v1
    dot = (v0 * v1).sum(-1, keepdim=True).clamp(-1.0, 1.0)

    # When vectors are nearly parallel, sin(theta) -> 0
    # Fall back to normalized linear interpolation
    close_condition = (torch.abs(dot) > 0.9995)
    linear_interp = F.normalize(v0 + t * (v1 - v0), dim=-1)

    # Standard slerp
    theta = torch.acos(dot)                                      # [B, 1]
    sin_theta = torch.sin(theta)                                 # [B, 1]

    # Paper: sin(alpha*theta)/sin(theta) * h'_i
    #      + sin((1-alpha)*theta)/sin(theta) * h'_j
    scale_i = torch.sin(t * theta) / sin_theta                  # for v0 = h'_i
    scale_j = torch.sin((1.0 - t) * theta) / sin_theta          # for v1 = h'_j

    slerp_interp = scale_i * v0 + scale_j * v1
    slerp_interp = F.normalize(slerp_interp, dim=-1)

    return torch.where(close_condition, linear_interp, slerp_interp)