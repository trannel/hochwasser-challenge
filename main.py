import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

# from analysis import *
from test import create_solution
from train import select_features, DataWindowDatasetFeatures, MLP, train

seed = 0xBEEF
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Params
batch_size = 32
epochs = 20
learning_rate = 0.001
data_dir = "data"  # Enter the path of your choice
train_filename = f"{data_dir}/x_train.csv"

# Check for a GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"GPU: '{device}' available" if device.type == "cuda" else f"No GPU available, pytorch will run on your CPU.")

raw_data = pd.read_csv(train_filename, sep=";", date_parser="date")

# # Exploration
# box_plotter(raw_data)
# describe(raw_data)
# data_check(raw_data)
# # Data visualization
# multi_singleplot(raw_data)
# test_pred = np.random.rand(12) + 18.5
# data_win(raw_data, 0, 1, pred=test_pred)

# # Feature extraction
# periodizer(raw_data, date_format='day')
# periodizer(raw_data, date_format='year')


# Load the Data into a Pytorch Dataset
data = select_features(raw_data)

# Scale the inputs
scaler = StandardScaler()
data[data.columns] = scaler.fit_transform(data[data.columns])

# train 80% & validation 20%
train_df = data.iloc[:int(len(data) * 0.8)]
val_df = data.iloc[int(len(data) * 0.8):]
assert len(train_df) == 26522, "The train data size should be 26522"
assert len(val_df) == 6631, "The validation data size should be 6631"
train_ds = DataWindowDatasetFeatures(df=train_df, input_size=48, input_dim=train_df.shape[1], pred_size=12)
val_ds = DataWindowDatasetFeatures(df=val_df, input_size=48, input_dim=val_df.shape[1], pred_size=12)

idx = 0
idx_input = 0
idx_target = 1
print(f"len train dataset: {len(train_ds)} shape input: {train_ds[idx][idx_input].shape} shape target {train_ds[idx][idx_target].shape}")


# Put Datasets into a Dataloader to get Batches
train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_ds, shuffle=False, batch_size=batch_size)

# init model
model = MLP(in_seq_len=48, in_dim=train_df.shape[1], out_seq_len=12, hidden_size=32).to(device)

# train
model = train(model=model, train_loader=train_loader, val_loader=val_loader, device=device, data_dir=data_dir,
              batch_size=batch_size, learning_rate=learning_rate, epochs=epochs)

# create solution
create_solution(data_dir, scaler, batch_size, model, device)
