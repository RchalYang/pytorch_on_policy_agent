import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Tensor(nparray):
    return torch.Tensor(nparray, ).to(device)

def Tensor_zeros_like(target):
    return torch.zeros_like(target).to(device)

# def 