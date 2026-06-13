import torch
import torch.nn.functional as F

def slerp(v0, v1, t=0.5):
    v0 = F.normalize(v0, dim=-1)
    v1 = F.normalize(v1, dim=-1)
    
    dot = (v0 * v1).sum(-1, keepdim=True).clamp(-1.0, 1.0)

    close_condition = (torch.abs(dot) > 0.9995)
    linear_interp = F.normalize(v0 + t * (v1 - v0), dim=-1)
    

    theta = torch.acos(dot)           
    sin_theta = torch.sin(theta)      
    
    scale_i = torch.sin(t * theta) / sin_theta       
    scale_j = torch.sin((1.0 - t) * theta) / sin_theta 
    
    slerp_interp = scale_i * v0 + scale_j * v1
    slerp_interp = F.normalize(slerp_interp, dim=-1)  
    
    return torch.where(close_condition, linear_interp, slerp_interp)