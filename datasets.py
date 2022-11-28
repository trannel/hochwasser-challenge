import pandas as pd
import torch
from torch import Tensor
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

# In this case, we will define a simple MLP which can be used to forecast all 12-time steps at once. This NN has only one layer, which should not perform well. Feel free to experiment with different layers, add new layers and use different NN architectures to improve your results.

# Create A Window Dataset
# class DataWindowDataset(Dataset):
#     """
#     This is a Pytorch Dataset. More Information can be found here: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
#     The dataset returns the rolling window dataset with input x [x_1, .., x_48] and the target y [y_1, .., y_12]
#     """
#     def __init__(self, df: pd.DataFrame, input_size: int = 48, pred_size: int = 12):
#         self.df = df
#         self.window_len = input_size + pred_size
#         self.pred_size = pred_size
#         self.input_size = input_size
#
#     def __len__(self) -> int:
#         return len(self.df) - self.window_len
#
#     def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
#         x = self.df.iloc[idx: idx + self.input_size].values
#         y = self.df.iloc[idx + self.input_size: idx + self.window_len].values
#
#         return torch.tensor(x, dtype=torch.float32).squeeze(), torch.tensor(y, dtype=torch.float32).squeeze()


class DataWindowDatasetFeatures(Dataset):
    def __init__(self, df: pd.DataFrame, input_size: int = 48, input_dim: int = 1, pred_size: int = 12):
        self.df = df
        self.window_len = input_size + pred_size
        self.pred_size = pred_size
        self.input_size = input_size
        self.input_dim = input_dim

    def __len__(self) -> int:
        return len(self.df) - self.window_len

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        x = self.df.iloc[idx: idx + self.input_size, :].values
        y = self.df.iloc[idx + self.input_size: idx + self.window_len, -1].values

        return torch.tensor(x, dtype=torch.float32).squeeze(), torch.tensor(y, dtype=torch.float32).squeeze()
