import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def f(x):
    return np.sin(x) + np.cos(2 * x) + 2


def gaussian(x, c, v):
    return torch.exp(-((x - c) ** 2) / (2 * v ** 2))

# def train_RBF()


if __name__ == '__main__':
    x = np.linspace(0, 10, 500)
    print(x.shape)
    y = f(x)

    train_data, test_data, train_labels, test_labels = train_test_split(x, y, test_size=0.9, random_state=43)

    # plt.plot(x, y, label='data', color='black')
    #
    # plt.legend()
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()
