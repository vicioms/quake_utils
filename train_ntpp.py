import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import datetime
from libs.sequences import SeismicSequence
from libs.iris import irisRequests
from libs.distributions import Weibull, WeibullMM, GaussianMM, InterTimeDistribution
from importlib import reload  # Python 3.4+
import matplotlib.pyplot as plt
from ntpp_utils import get_time_nll_loss


def load_hauksson():
    column_names = ["year","month","day", "hour", "minut", 'sec', "id",
                    "lat","lon", "depth", "mag", "nPhases", "azGap", "nearDist", "hErr", "vErr", "residual", "flag1", "flag2"]
    df = pd.read_csv("catalogs/sc_1981_2022q1_1d_3d_gc_soda.gc", sep="\s+", usecols=range(0,11), header=None,names=column_names[:11])
    df['time'] = pd.to_datetime(dict(year=df.year, month=df.month, day=df.day, hours=df.hour, minutes=df.minut, seconds=df.sec))
    return df