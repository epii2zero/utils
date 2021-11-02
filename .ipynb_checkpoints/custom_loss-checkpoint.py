import torch

eps = 1e-8

def cal_SDR_loss(original, estimate):
    """
    Training target: SDR_loss
    Args:
        original: torch.Tensor  [B, 1, T]  B: batch_size, T: #samples
        estimate: torch.Tensor  [B, 1, T]  B: batch_size, T: #samples
    Returns:
        SDR_loss: torch.Tensor [1]
    """
    # 0. Unsqueeze
    original = original.squeeze(dim=1)
    estimate = estimate.squeeze(dim=1)
    
    # original 
    # Dot product
    dot_product = torch.sum(original*estimate, dim=1, keepdim=True) # [B, 1]
    power_dot   = dot_product ** 2  # [B, 1]
    # Power
    power_est   = torch.sum(estimate**2, dim=1, keepdim=True) + eps  # [B, 1]
    # SDR loss
    loss = 0 - power_dot / power_est

    return torch.mean(loss)