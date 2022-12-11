import os

from matplotlib import pyplot as plt
from torch import nn, optim

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List


# Create A Window Dataset
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


class MLP(nn.Module):
    def __init__(self, in_seq_len: int = 48, in_dim: int = 1, out_seq_len: int = 12, hidden_size: int = 64, dropout_p: float = 0.1):
        """
        This is a simple MLP which can be used as a starting point to predict a sequence. Different layers can be added in self.layers.
        @param in_seq_len: The input sequence length x_0, .., x_n  n = input sequence length
        @param in_dim: The input feature dimension
        @param out_seq_len: The output sequence length y_0, .., y_m  m = target sequence length
        @param hidden_size: The dimension of the hidden layers
        @param dropout_p: dropout probability if a dropout is used
        """
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.in_dim = in_dim

        self.layers = nn.Sequential(
            # nn.Sigmoid(),
            # nn.Linear(in_seq_len * in_dim, out_seq_len),
            nn.Linear(in_seq_len * in_dim, hidden_size),
            # nn.Linear(hidden_size, hidden_size),
            # nn.Tanh(),
            nn.Linear(hidden_size, out_seq_len),
        )


    def forward(self, x: torch.tensor) -> Tensor:
        """
        This forward method is called by using model(x). It uses the input x to predict all time steps, so that the output
        should be the length od the target sequence
        @param x: input of the NN
        @return: output of the NN
        """
        batch_size = x.shape[0]
        if self.in_dim == 1:
            assert len(x.shape) == 2, f"if only one input feature is provided the input must have the input shape " \
                                      f"(batch_size, seq_len), but the shape is {x.shape}"
        else:
            assert len(x.shape) == 3 and self.in_dim == x.shape[2], f"if more than one input feature is provided the input must have " \
                                                                    f"the input shape (batch_size, seq_len, in_dim), but the shape is {x.shape}"

        return self.layers(x.reshape(batch_size, -1))


def train(model: MLP, train_loader: DataLoader, val_loader: DataLoader, device, data_dir, batch_size: int,
          learning_rate: float = 0.001,
          epochs: int = 2) -> MLP:
    """
    Train Loop
    @param model: The NN Model. The model must have the same forward return shape as the dataset has as target
    @param train_loader: Loader with training data
    @param val_loader: Loader with validation data
    @param learning_rate: Learning rate for the Adam optimizer
    @param epochs: number of training runs
    """
    loss_fct = nn.MSELoss()
    # initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    size_train_loader = len(train_loader.dataset)
    train_mse = []
    val_mse = []
    for epoch_i in range(epochs):
        train_losses = []
        for batch_i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # forward pass, compute predicted y
            pred = model(x)
            loss = loss_fct(pred, y)
            train_losses.append(loss.item())

            # backpropagation part
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            if batch_i % 200 == 0:
                loss, current = loss.item(), batch_i * batch_size
                print(
                    f"{epoch_i:>2d} - [{current:>5d}/{size_train_loader:>5d}] MSE loss: {loss:>5f} ")

        # validate on data which was not used to train the model
        val_losses = []
        with torch.no_grad():
            for batch_i, (x, y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss = loss_fct(pred, y)
                val_losses.append(val_loss.item())

        train_losses = np.mean(np.array(train_losses))
        val_losses = np.mean(np.array(val_losses))
        print(
            "-" * 20 + "\n" +
            f"Tra MSE Loss {train_losses} | Tra RMSE Loss {np.sqrt(train_losses)}\n" +
            f"Val MSE Loss {val_losses} | Val RMSE Loss {np.sqrt(val_losses)}\n" +
            "-" * 20 + "\n"
        )
        train_mse.append(train_losses)
        val_mse.append(val_losses)
    plot_loss(train_mse, val_mse, data_dir)
    return model


def plot_loss(train_mse: List[np.ndarray], val_mse: List[np.ndarray], data_dir: str):
    fig, axs = plt.subplots(2)
    fig.suptitle('Model loss')
    axs[0].plot(train_mse, label="train")
    axs[0].plot(val_mse, label="validation")
    axs[0].set_ylabel('MSE loss')
    axs[0].set_xlabel('epoch')
    axs[1].plot(np.sqrt(train_mse), label="train")
    axs[1].plot(np.sqrt(val_mse), label="validation")
    axs[1].set_ylabel('RMSE loss')
    axs[1].set_xlabel('epoch')
    fig.legend(['train', 'validation'], loc='upper left')
    fig.savefig(get_path(f"{data_dir}/model_loss.png"))

def get_path(path: str) -> str:
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = filename + str(counter) + extension
        counter += 1
    return path

def select_features(data: pd.DataFrame):
    data.date = pd.to_datetime(data.date)
    data.index = data.date
    data = data.drop(columns=['date'], axis=0)

    data['year_sin'] = np.sin(data.index.hour * 2 * np.pi / 365)
    data['day_sin'] = np.sin(data.index.dayofyear * 2 * np.pi / 24)
    data['week_sin'] = np.sin(data.index.dayofweek * 2 * np.pi / 7)
    cols = [
        "Discharge,  Stausee Beyenburg",
        "Water Level,  Stausee Beyenburg",
        "Precipitation,  Beyenburg",
        "Water Level,  Leimbach",
        # "Precipitation,  Barmen Wupperverband Hauptverwaltung",
        "year_sin",
        "day_sin",
        "week_sin",
        "Water Level,  Kluserbrücke",
    ]
    data = data[cols]
    assert 'year_sin' in data.columns and "Water Level,  Kluserbrücke" in data.columns, "data must include the columns year_sin and Water Level,  Kluserbrücke"
    return data
