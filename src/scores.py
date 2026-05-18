import torch

def pure_weight_grad(param):
    return torch.abs(param.grad)

def first_order_taylor_weight_grad(param):
    # Score = | Weight * Gradient |
    return torch.abs(param * param.grad)