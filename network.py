import torch
from torch import Tensor
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

# Train Loop
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
            # nn.Linear(in_seq_len * in_dim, hidden_size),
            # nn.Sigmoid(),
            # nn.Linear(hidden_size, out_seq_len),
            nn.Linear(in_seq_len * in_dim, out_seq_len),
            # nn.Linear(in_seq_len * in_dim, hidden_size),
            # nn.Tanh(),
            # nn.Linear(hidden_size, out_seq_len)
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