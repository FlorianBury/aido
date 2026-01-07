from typing import List, Tuple, Self

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from aido.logger import logger

class SurrogateDataset(Dataset):
    def __init__(self):
        pass

    def load(self,filepath):
        out = torch.load(
            filepath,
            weights_only = False,
            map_location = torch.device('cpu'),
        )
        data = out['data']
        slices = out['slices']
        assert "particles" in data.node_types
        assert "particles" in data.node_types

class Surrogate(nn.Module):
    pass
