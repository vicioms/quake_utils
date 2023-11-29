import torch
import numpy as np


class InterTimeDistribution:
    def __init__(self) -> None:
        pass
    
    
    def get_log_prob(self, x) -> None:
        pass

    def get_log_survival(self, x) -> None:
        pass
    
class Weibull(InterTimeDistribution):
    
    def __init__(self, b_params, k_params,eps=1e-8) -> None:
        super().__init__()
        self.b = b_params
        self.k = k_params
        self.eps = eps
        
        
    def get_log_prob(self, x) -> torch.Tensor:
        # x must have shape equal to b & k
        # assure no inter_time is zero
        x = x.clamp_min(self.eps)
        return (self.b.log() + self.k.log() + (self.k - 1) * x.log() 
                + self.b.neg() * torch.pow(x, self.k))
        

    def get_log_survival(self, x) -> torch.Tensor:
        x = x.clamp_min(self.eps)
        return self.b.neg() * torch.pow(x, self.k)
    
    
class WeibullMM(InterTimeDistribution):
    
    def __init__(self, pi_params, b_params, k_params,eps=1e-8) -> None:
        super().__init__()
        # pi_params, b_params, k_params  must be (N, L, R) where R is the number of models in the mixture
        self.pip = pi_params
        self.b = b_params
        self.k = k_params
        self.eps = eps
        
        
    def get_log_prob(self, x) -> torch.Tensor:
        # x must have shape equal to b & k
        # assure no inter_time is zero
        x = x.clamp_min(self.eps)
        n_mixtures = self.b.size()[-1]
        x_rep = x[:,:,None].repeat((1,1,n_mixtures))
        #x = torch.unsqueeze(x, -1)
        log_p = (self.b.log() + self.k.log() + (self.k - 1) * x_rep.log() 
                + self.b.neg() * torch.pow(x_rep, self.k))
                
        return torch.sum(log_p*self.pip, dim=-1)

    def get_log_survival(self, x) -> torch.Tensor:
        x = x.clamp_min(self.eps)
        n_mixtures = self.b.size()[-1]
        x_rep = x[:,:,None].repeat((1,1,n_mixtures))
        #x = torch.unsqueeze(x, -1)
        log_s = self.b.neg() * torch.pow(x_rep, self.k)
        return torch.sum(log_s*self.pip, dim=-1)
    
    
class GaussianMM:
    def __init__(self, pi_params, mu_params, tau_params) -> None:
        self.pip = pi_params
        self.mu = mu_params
        self.tau = tau_params
        
    def get_log_prob(self, x):
        x = torch.unsqueeze(x, -1)
        log_p = (-0.5)*self.tau.log() - 0.5*(x*x*self.tau)
        return torch.sum(log_p*self.pip, dim=-1)
    
        

class Gaussian_ale_easy(InterTimeDistribution):
    def __init__(self, mu_params, sigma_params) -> None:
        self.mu = mu_params
        self.sigma = sigma_params
        
    def get_log_prob(self, x):
        ones_tensor = torch.ones_like(x)
        return (-0.5) * torch.pow(torch.div(x + self.mu.neg(), self.sigma), 2) + self.sigma.log().neg() + 0.5*torch.log(ones_tensor*(2*torch.pi)).neg()