import torch 
import torch.nn as nn
import torch.nn.functional as F


class NMFBase(nn.Module):
  def __init__(self, 
               W_shape, 
               H_shape, 
               n_components, 
               initial_components=None,
               fix_components=(),
               initial_weights=None,
               fix_weights=()
               ):
    super().__init__()
    self.fix_neg = nn.Threshold(0.0, 1e-8)

                 
