import math
import torch
import torch.nn.functional as F


def stage2_loss(c, z, logit_scale):

    safe_logit_scale = torch.clamp(
        logit_scale.float(),
        min=0.0,
        max=math.log(100.0)
    )

    temperature = torch.exp(safe_logit_scale)

    logits = (c.float() @ z.float().T) * temperature

    labels = torch.arange(
        len(c),
        device=c.device
    )

    loss_q2t = F.cross_entropy(
        logits,
        labels
    )

    loss_t2q = F.cross_entropy(
        logits.T,
        labels
    )

    return (loss_q2t + loss_t2q) / 2.0