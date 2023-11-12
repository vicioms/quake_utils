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


class Gaussian(InterTimeDistribution):
    def __init__(self, mu_params, sigma_params) -> None:
        self.mu = mu_params
        self.sigma = sigma_params
        
    def get_log_prob(self, x):
        ones_tensor = torch.ones_like(x)
        return (-0.5) * torch.pow(torch.div(x + self.mu.neg(), self.sigma), 2) + self.sigma.log().neg() + 0.5*torch.log(ones_tensor*(2*torch.pi)).neg()
    

class Gaussian_xy(InterTimeDistribution):
    def __init__(self, meanx, meany, varx, vary, cov) -> None:
        self.mean_x = meanx
        self.mean_y = meany
        self.var_x = varx
        self.var_y = vary
        self.cov = cov
    
    def get_log_prob(self, pos_lat, pos_lon):
        arg = -0.5*torch.div((self.var_y * torch.pow(pos_lat, 2) + pos_lon*(-2*self.cov*pos_lat + self.var_x*pos_lon)), -torch.pow(self.cov, 2)+self.var_x*self.var_y)
        det = -0.5*(-torch.pow(self.cov, 2) + self.var_x*self.var_y)
        return arg + det
        
class Gaussian_xy_mixture(InterTimeDistribution):
    def __init__(self, means_mix, variances_mix, covariances_mix, Pi_params, n_mixtures) -> None:
        self.mean_mix = means_mix
        self.variances_mix = variances_mix
        self.covariances_mix = covariances_mix
        self.Pi_params = Pi_params
        self.means_x = means_mix[...,0:n_mixtures]
        self.means_y = means_mix[...,n_mixtures::]
        self.vars_x = variances_mix[...,0:n_mixtures]
        self.vars_y = variances_mix[...,n_mixtures::]
        self.covs = covariances_mix
        self.pip = Pi_params
    
    def get_log_prob(self, pos_lat, pos_lon):
        arg = -0.5*torch.div((self.vars_y * torch.pow(pos_lat, 2) + pos_lon*(-2*self.covs*pos_lat + self.vars_x*pos_lon)), -torch.pow(self.covs, 2)+self.vars_x*self.vars_y)
        det = -0.5*(-torch.pow(self.covs, 2) + self.vars_x*self.vars_y)
        return torch.sum((arg + det)*self.pip, dim=-1)

class Exponential(InterTimeDistribution):
    def __init__(self, lambda_param) -> None:
        self.lambda_coeff = lambda_param
        
    def get_log_prob(self, x):
        return self.lambda_coeff.log() - self.lambda_coeff*x
    

    
    


        
        