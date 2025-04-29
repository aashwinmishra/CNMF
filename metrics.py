import torch 
import torch.nn.functional as F


def kl_divergence(pred, target):
  return torch.sum(target * torch.log(target / pred)) + torch.sum(pred) - torch.sum(target) 


def euclidean_distance(pred, target):
  return F.mse_loss(pred, target, reduction='sum') / 2


def beta_divergence(pred, target, beta = 1):
  if beta == 1:
    return kl_divergence(pred, target)
  elif beta == 2:
    return euclidean_distance(pred, target)

