from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import os

from network import MLP


# Create a window that shows 24 hours of observed data and the right dimensions to leave some space for the 6-hour prediction that is the task of this challenge. Ideally the function is flexible to be used easily on different intervals of our data.
def data_win(data: pd.DataFrame, start: int, column: int, pred=None):
    """
    Function that takes in a dataframe of sensor readings (data), an index (sorted by time) to start from and a column of the
    dataframe, given in as column number (0 to x). It returns a plot with 48 entries (24 hours)of observed data, 12 entries
    (6 hours) of observed that had to be predicted and, in case that own predictions were fed in, a visualization of the 12
    predictions entries.
    """

    windata = data.iloc[start:start + 48, column]
    labels = data.iloc[start + 48:start + 48 + 12, column]

    if pred is not None:
        plt.plot(np.arange(start + 48, start + 60, 1), pred, label='Prediction')

    plt.plot(np.arange(start, start + 48, 1), windata, label='Observed')
    plt.plot(np.arange(start + 48, start + 60, 1), labels, label='Target')

    plt.xlabel('Time [2/h)]')
    plt.ylabel(f'{data.keys()[column]}')
    plt.legend()

    plt.show()


# To predict the water level at the 'Kluserbrücke' measuring station, the simplest model would only try to predict future values by looking at previously measured water levels (the univariate case).
# As there are other inputs available, like water levels at different measuring stations or precipitation measurements they could obviously be included in any model as well.
# Besides these other directly available sensor inputs it could possibly be worthwhile to consider feature extraction/engineering in this case.
# As an easy example one could use the time, shaped into a feature that especially depicts the periodic character of time (there is periodicity in time as we look at it day- or year-wise for example).
# Example: Day-wise periodic feature:

# This function creates two day-wise periodic features (sin and cos) and adds those as columns to the dataframe.
# It also shows a little plot as demonstration.
# Expand the function to also accord for two year-wise periodic features
def periodizer(data: pd.DataFrame, date_format: str = 'day'):
    """
    Function to create a daily or yearly periodic feature out of a given dataset with a readable timestamp.
    date_format takes in a string (either 'day' or 'year') to define what feature has to be created.
    Output is an exemplary presentation of the periodic feature,
    with x-axis = time [h] and y-axis shows a sine or cosine signal.
    """

    per_dataset = data

    if date_format == 'day':
        per_data = per_dataset.index.hour
        # Day has 24 hours
        day = 24
        per_dataset['Day_sin'] = np.sin(per_data * 2 * np.pi / day)
        per_dataset['Day_cos'] = np.cos(per_data * 2 * np.pi / day)

        # Show a small example of how the signal looks
        plt.plot(np.array(per_dataset['Day_sin'])[0:200])
        plt.plot(np.array(per_dataset['Day_cos'])[0:200])

        plt.show()
        # return(per_dataset)

    elif date_format == 'year':
        per_data = per_dataset.index.day_of_year

        # Year has 365 days
        year = 365

        per_dataset['Year_sin'] = np.sin(per_data * 2 * np.pi / year)
        per_dataset['Year_cos'] = np.cos(per_data * 2 * np.pi / year)
        # Show a small example of how the signal looks
        plt.plot(np.array(per_dataset['Year_sin'])[0:20000])
        plt.plot(np.array(per_dataset['Year_cos'])[0:20000])

        plt.show()

    else:
        print("Incorrect date_format given in. Has to be either 'day' or 'year'.")
    return ()

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
    train_mse= []
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
    # fig.show()
    fig.savefig(get_path(f"{data_dir}/model_loss.png"))


def select_features(data: pd.DataFrame):
    data.date = pd.to_datetime(data.date)
    data.index = data.date
    data = data.drop(columns=['date'], axis=0)

    # data = data["Water Level,  Kluserbrücke"].to_frame()
    data['year_sin'] = np.sin(data.index.hour * 2 * np.pi / 365)
    data['day_sin'] = np.sin(data.index.dayofyear * 2 * np.pi / 24)
    data['week_sin'] = np.sin(data.index.dayofweek * 2 * np.pi / 7)
    # print("input features:" + data.columns, len(data.columns))
    # exit()
    assert 'year_sin' in data.columns and "Water Level,  Kluserbrücke" in data.columns, "data must include the columns year_sin and Water Level,  Kluserbrücke"
    data = data[["year_sin", "Water Level,  Kluserbrücke"]]
    return data


def get_path(path: str) -> str:
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = filename + str(counter) + extension
        counter += 1
    return path
