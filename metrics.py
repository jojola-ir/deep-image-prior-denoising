import torch

def PSNR(input, target):
    return -10*torch.log10(torch.mean((input - target) ** 2, dim=[1, 2, 3])+EPS)

def MSE(input, target):
    return torch.mean((input - target) ** 2, dim=[1, 2, 3])