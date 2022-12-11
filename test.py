import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import Dataset, DataLoader

from train import select_features, get_path


# Create a Test Pytorch Dataset
class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, input_size=48, input_dim=1, pred_size=12):
        self.df = df
        self.window_len = input_size + pred_size
        self.pred_size = pred_size
        self.input_size = input_size
        self.input_dim = input_dim

    def __len__(self) -> int:
        return int(len(self.df) / self.input_size)

    def __getitem__(self, idx: int) -> Tensor:
        x = self.df.iloc[idx * self.input_size: (idx + 1) * self.input_size, :].values
        x = torch.tensor(x, dtype=torch.float32)
        if self.input_dim == 1:
            x = x.reshape(-1)
        return x

# Test Loop
# This function loops through the test data and saves the prediction in the test_value list so that the results can be uploaded to the challenge.
def test(model: nn.Module, test_loader: DataLoader, device) -> np.array:
    test_values = []
    with torch.no_grad():
        for batch_i, x in enumerate(test_loader):
            x = x.to(device)
            pred = model(x)
            test_value_batch = pred.cpu().detach().numpy()
            # loop over the batches to add the prediction to the test_values in the right order
            for test_prediction in test_value_batch:
                for test_value in test_prediction:
                    test_values.append(test_value)

    return np.array(test_values)


def create_solution(data_dir: str, scaler, batch_size: int, model, device):
    # load test data
    test_data = pd.read_csv(f"{data_dir}/x_test.csv", sep=";", date_parser="date")
    test_data = select_features(test_data)
    test_data[test_data.columns] = scaler.transform(test_data)

    test_ds = TestDataset(test_data, input_dim=test_data.shape[1])
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=batch_size)

    test_values = test(model=model, test_loader=test_loader, device=device)
    assert len(test_values) == 1116, "The prediction must have the length 1116, otherwise values are missing or there are too many values"

    # Scaler Inverse Scale the predicted values
    features = scaler.n_features_in_
    shape = (features-1, test_values.shape[0])
    lists = list(np.zeros(shape))
    lists.append(test_values)
    scale_array = np.array(lists)
    test_values = scaler.inverse_transform(scale_array.T)[:, features-1]

    # Load Sample file
    y_test_data = pd.read_csv(f"{data_dir}/y_test.csv")
    y_test_data.prediction = test_values
    print(y_test_data.head())

    # Save to CSV file
    path = get_path(f"{data_dir}/solution.csv")

    y_test_data.to_csv(path, index=False)
