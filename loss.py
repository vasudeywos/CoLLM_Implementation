import torch
import torch.nn.functional as F
import math

def collm_loss(c_v, c_w, c, z, logit_scale):
    safe_logit_scale = torch.clamp(logit_scale.float(), min=0.0, max=math.log(100.0))
    temperature = torch.exp(safe_logit_scale)

    def contrastive_loss(query, target):
        query = query.float()
        target = target.float()
        temp = temperature.float()          # ← cast here, not at top
        logits = (query @ target.T) * temp
        labels = torch.arange(len(query), device=query.device)
        loss_q2t = F.cross_entropy(logits, labels)
        loss_t2q = F.cross_entropy(logits.T, labels)
        return (loss_q2t + loss_t2q) / 2.0

    loss_v = contrastive_loss(c_v, z)
    loss_w = contrastive_loss(c_w, z)
    loss_c = contrastive_loss(c, z)
    total = (loss_v + loss_w + loss_c) / 3.0

    return total, {
        "loss": total.item(),
        "img_only": loss_v.item(),
        "txt_only": loss_w.item(),
        "comp":     loss_c.item(),
    }
