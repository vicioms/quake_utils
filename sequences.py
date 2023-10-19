import torch
import pandas as pd
import numpy as np
import datetime
from typing import Union, Optional

# inspired by https://zenodo.org/record/8161777
class SeismicSequence:
    @staticmethod
    def arrival_to_inter_times(arrival_times : Union[torch.tensor, np.ndarray, list],t_start : float, t_end : float):
        if isinstance(arrival_times, torch.Tensor):
            return torch.diff(arrival_times,prepend=t_start, append=t_end)
        else:
            return np.diff(arrival_times,prepend=t_start, append=t_end)
    
    @staticmethod
    def pack_sequences(s_list : list):
        raise NotImplementedError()

    def __init__(self, inter_times: Union[torch.Tensor, np.ndarray],
                 t_start : float = 0.0,
                 t_nll_start : Optional[float] = None,
                 features : Optional[Union[torch.Tensor, np.ndarray]] = None):
    
        self.inter_times =  torch.flatten(torch.as_tensor(inter_times))
        if not self.inter_times.dtype in [torch.float32, torch.float64]:
            raise ValueError(
                f"Supported types for inter_times are torch.float32 or torch.float64"
                "(got {self.inter_times.dtype})"
            )
            
        self.arrival_times = self.inter_times.cumsum(dim=-1)[:-1] + t_start
        self.t_start = float(t_start)
        self.t_end = float(self.inter_times.sum().item() + self.t_start)
        if t_nll_start is None:
            t_nll_start = t_start
        self.t_nll_start = float(t_nll_start)
        if(features is not None):
            self.features = torch.as_tensor(features)
            if self.features.shape[0] != self.arrival_times.shape[0]: # different sequence length!!!!
                raise ValueError(
                f"The length of features is different from the (induced) length of arrival times."
            )
        else:
            self.features = None
            
    def get_subsequence(self,
                        start : float,
                        end : float):
        if start < self.t_start or end > self.t_end:
            raise ValueError(
                f"Error in either start or end. start must be >= {self.t_start} and end must be <= {self.t_end}"
            )
        mask = (self.arrival_times >= start) & (self.arrival_times <= end)
        new_arrival_times = self.arrival_times[mask]
        
        
        
        
        new_arrival_times = self.arrival_times[mask]
        if(len(new_arrival_times) > 0):
            # find the last gap, to add to the new inter_times
            # we keep type and device
            last_inter_time = torch.tensor(
                [end - new_arrival_times[-1]],
                device=self.inter_times.device,
                dtype=self.inter_times.dtype,
            )
            # we skip the last element from inter_times, as it refers to the old last time
            new_inter_times = torch.cat([self.inter_times[:-1][mask], last_inter_time])
            first_inter_time = new_arrival_times[0] - start
            new_inter_times[0] = first_inter_time
            print(new_inter_times)
            print(new_arrival_times)
