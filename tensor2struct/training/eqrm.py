import attr
import math
import torch
import torch.nn as nn
import torch.autograd as autograd
import collections

import copy
import higher
import logging

from tensor2struct.modules.distributions import Nonparametric

logger = logging.getLogger("tensor2struct")


class EQRM(nn.Module):
    def __init__(self, device=None, 
                quantile=0.75, 
                burnin_iters=2500, 
                dist=None):
        super().__init__()
        self.device = device
        self.quantile = quantile
        self.burnin_iters = burnin_iters
        
        # self.register_buffer('update_count', torch.tensor([0]))
        self.register_buffer('alpha', torch.tensor(self.quantile, dtype=torch.float64))

        if dist is None:
            self.dist = Nonparametric()
        else:
            self.dist = dist
    
    def train(self, model, batch, step):
        assert model.training
        
        ret_dic = {}
        with torch.set_grad_enabled(model.training):
          "1. Feed data to encoder"
          enc_list = [enc_input for enc_input, dec_output in batch]
          enc_states = model.encoder(enc_list)
          
          "2. For enc in encoded"
          losses = []  # list cannot maintain grad_fn
          for enc_state, (enc_input, dec_output) in zip(enc_states, batch):
              ret_dic = model.decoder(dec_output, enc_state)
              losses.append(ret_dic["loss"])

        return losses
     
    def transform(self, losses, step, unlabeled=None):
        losses = torch.cat([loss.reshape(1) for loss in losses])
        
        if step < self.burnin_iters:
            # Burn-in/anneanlig period uses ERM like penalty methods
            loss = torch.mean(losses)
        else:
            # Loss is the alpha-quantile value
            self.dist.estimate_parameters(losses)
            loss = self.dist.icdf(self.alpha)
        
        reset_opt = True if step == self.burnin_iters else False  # Reset optimizer at burnin_iters step
        return loss, reset_opt
    