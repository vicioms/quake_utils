import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.sequences import SeismicSequence
from libs.distributions import Weibull, WeibullMM, GaussianMM, InterTimeDistribution

def get_time_nll_loss(inter_times : torch.Tensor, seq_lengths : torch.Tensor, inter_time_distr : InterTimeDistribution):
    log_prob = inter_time_distr.get_log_prob(inter_times)
    mask = SeismicSequence.get_lengths_mask(seq_lengths, inter_times.shape[1])
    log_like = (log_prob * mask).sum(-1)  # (N,)
    log_surv = inter_time_distr.get_log_survival(inter_times)  # (N, L)
    end_idx = torch.unsqueeze(seq_lengths,-1)  # (N, 1)
    log_surv_last = torch.gather(log_surv, dim=-1, index=end_idx)  # (N, 1)
    log_like += log_surv_last.squeeze(-1)  # (N,)
    return -log_like


def get_spatial_nll_loss( locations : torch.Tensor,   seq_lengths : torch.Tensor, spatialDistr):
    log_prob = spatialDistr.get_log_prob(locations)
    