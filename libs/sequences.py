import torch
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import datetime
from typing import Union, Optional

# inspired by https://zenodo.org/record/8161777
class SeismicSequence:

    @staticmethod
    def from_pandas_df(df, time_label = "Time", other_labels = ["Latitude", "Longitude", "Magnitude"], unit='s'):
        t_start = df[time_label].iloc[0].to_datetime64().astype('datetime64[D]')
        t_end = df[time_label].iloc[-1].to_datetime64().astype('datetime64[D]') + np.timedelta64(1, 'D')
        arrival_times = df[time_label].values
        inter_times = np.diff(arrival_times, prepend=t_start, append=t_end)
        if(unit == 'D'):
            inter_times = inter_times.astype('float')*1e-9/86400
        elif(unit == 's'):
            inter_times = (inter_times.astype('float')*1e-9)
        else:
            raise NotImplementedError("Time unit %s is not implemented." % unit)        
        features = df[other_labels].values
        return SeismicSequence(inter_times, t_start=0, features=features)
    @staticmethod
    def arrival_to_inter_times(arrival_times : Union[torch.tensor, np.ndarray, list],t_start : float, t_end : float):
        if isinstance(arrival_times, torch.Tensor):
            return torch.diff(arrival_times,prepend=t_start, append=t_end)
        else:
            return np.diff(arrival_times,prepend=t_start, append=t_end)
    
    @staticmethod
    def pack_sequences(seq_list : list):
        inter_times = [seq.inter_times for seq in seq_list if isinstance(seq, SeismicSequence)]
        inter_times = pad_sequence(inter_times, batch_first=True, padding_value=0)
        features = [seq.features for seq in seq_list if isinstance(seq, SeismicSequence)]
        lengths = torch.as_tensor([len(f) for f in features])
        features = pad_sequence(features, batch_first=True, padding_value=0 )
        return inter_times, features, lengths
    
    @staticmethod
    def get_lengths_mask(lengths, longest_sequence_length):
        lengths_arange = torch.arange(longest_sequence_length, device=lengths.device)
        mask = (lengths_arange[None, :] < lengths[:, None]).float()  # (N, longest_sequence_length)
        return mask

    
        
    def __init__(self, inter_times: Union[torch.Tensor, np.ndarray],
                 t_start : float = 0.0,
                 t_nll_start : Optional[float] = None,
                 features : Optional[Union[torch.Tensor, np.ndarray]] = None):
        # we save the inter arrival times ("tau")
        self.inter_times =  torch.flatten(torch.as_tensor(inter_times, dtype=torch.float32))
        if not self.inter_times.dtype in [torch.float32, torch.float64]:
            raise ValueError(
                f"Supported types for inter_times are torch.float32 or torch.float64"
                "(got {self.inter_times.dtype})"
            )
        # the arrival times  ("t") are obtained from inter_times  by summing them and adding t_start
        self.arrival_times = self.inter_times.cumsum(dim=-1)[:-1] + t_start
        self.t_start = float(t_start)
        self.t_end = float(self.inter_times.sum().item() + self.t_start)
        if t_nll_start is None:
            t_nll_start = t_start
        self.t_nll_start = float(t_nll_start)
        if(features is not None):
            self.features = torch.as_tensor(features, dtype=torch.float32)
            if self.features.shape[0] != self.arrival_times.shape[0]: # different sequence length!!!!
                raise ValueError(
                f"The length of features is different from the (induced) length of arrival times."
            )
        else:
            self.features = None
            
    def get_subsequence(self, start : float, end : float):
        if start < self.t_start or end > self.t_end:
            raise ValueError(
                f"Error in either start or end. start must be >= {self.t_start} and end must be <= {self.t_end}"
            )
        mask = (self.arrival_times >= start) & (self.arrival_times <= end)
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
        else:
            new_inter_times = torch.tensor(
                [end - start],
                device=self.inter_times.device,
                dtype=self.inter_times.dtype,
            )
        new_features = self.features[mask].contiguous()
        return SeismicSequence(new_inter_times,
                               t_start=start,
                               t_nll_start=max(self.t_nll_start, start),
                               features=new_features)