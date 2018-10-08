import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Tensor(nparray):
    return torch.Tensor(nparray, ).to(device)

def Tensor_zeros_like(target):
    return torch.zeros_like(target).to(device)

# def 


def gaussian_neg_log_prob(mean, std, tensor_array):
    '''
    calculate the negative log probability of tensor_array given mean and std

    mean, std, numpy_array: n * d pytorch tensor matrix, n is the number of input, d is the dimension of action
    '''
    temp = (tensor_array - mean) / std
    return 0.5 * torch.sum(temp * temp, 1) \
           + 0.5 * np.log(2.0 * np.pi) * tensor_array.size()[1] \
           + torch.sum(torch.log(std), 1)


def gaussian_log_prob(mean, std, tensor_array):
    return -gaussian_neg_log_prob(mean, std, tensor_array)