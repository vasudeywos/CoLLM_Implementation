import torch
import torch.nn.functional as F

def collm_loss(c_v, c_w, c, z, logit_scale):
    """
    CoLLM Stage-1 training loss — Paper Equation 6
    Uses a learnable temperature (logit_scale) instead of fixed 0.07.
    """
    # Clamp temperature to avoid numerical instability
    temperature = torch.clamp(torch.exp(logit_scale), max=100.0)
    
    def contrastive_loss(query, target):
        logits = (query @ target.T) * temperature
        labels = torch.arange(len(query), device=query.device)
        loss_q2t = F.cross_entropy(logits, labels)
        loss_t2q = F.cross_entropy(logits.T, labels)
        return (loss_q2t + loss_t2q) / 2.0

    loss_v = contrastive_loss(c_v, z)
    loss_w = contrastive_loss(c_w, z)
    loss_c = contrastive_loss(c, z)
    
    total = (loss_v + loss_w + loss_c) / 3.0
    
    return total, {
        "loss_total": total.item(),
        "loss_image_only": loss_v.item(),
        "loss_text_only": loss_w.item(),
        "loss_composed": loss_c.item(),
    }