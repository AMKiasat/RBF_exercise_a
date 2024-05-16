import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class RBF(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_centroids, centers, wights, bias, variance):
        super(RBF, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(num_centroids, output_dim)
        self.centers = nn.Parameter(torch.Tensor(centers).float())
        self.linear.weight = nn.Parameter(torch.Tensor(wights).float())
        self.linear.bias = nn.Parameter(torch.Tensor(bias).float())
        self.variance = nn.Parameter(torch.ones(num_centroids))

    def gaussian(self, x, c, sigma):
        return torch.exp(-((x - c) ** 2) / (2 * sigma ** 2))

    def forward(self, x):
        with torch.no_grad():
            basis_func_output = self.gaussian(
                x.unsqueeze(1), self.centers.unsqueeze(0), self.variance
            )
            # Compute hidden layer output
            hidden_output = basis_func_output.sum(dim=2)
            # Compute final output
            output = self.linear(hidden_output)
            return output


if __name__ == '__main__':
    x = [2, 4, 5, 7, 8]
    y = [1.0, 2.1, 2.5, 3.6, 4.2]
    centers = [[3.0], [6.0], [9.0]]
    w = [[-0.5, -0.3, 0.8]]
    b = [0.1]
    v = [3]

    input_dim = 1
    hidden_dim = 2
    output_dim = 1
    num_centroids = 3

    model = RBF(input_dim, hidden_dim, output_dim, num_centroids, centers, w, b, v)

    outputs = model.forward(torch.Tensor(torch.tensor(x).unsqueeze(1)))
    criterion = nn.MSELoss()
    loss = criterion(torch.Tensor(torch.tensor(y).unsqueeze(1)), outputs)
    print(outputs)
    # for i in range(len(x)):
    #