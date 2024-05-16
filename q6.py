import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def f(x):
    return np.sin(x) + np.cos(2 * x) + 2


def gaussian(x, c, v):
    return np.exp(-((x - c) ** 2) / (2 * v ** 2))


def train_RBF(x, y, center_num):
    d = []

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            d.append(np.linalg.norm(x[i] - x[j]))
    variance = np.sort(d)[int(len(x) * (len(x) / 2) / 2)]
    # print(variance)

    centers = np.random.choice(x, size=center_num)
    g_distances = np.zeros((len(x), center_num))

    for i in range(len(x)):
        for j in range(len(centers)):
            g_distances[i, j] = gaussian(x[i], centers[j], variance)
    wights = np.dot(np.linalg.pinv(g_distances), y)

    return wights, centers, variance


def test_RBF(x, wights, centers, variance):
    g_distances = np.zeros((len(x), len(centers)))
    for i in range(len(x)):
        for j in range(len(centers)):
            g_distances[i, j] = gaussian(x[i], centers[j], variance)
    return np.dot(g_distances, wights)


if __name__ == '__main__':
    x = np.linspace(0, 10, 500)
    y = f(x)
    c_num = 12

    train_data, test_data, train_labels, test_labels = train_test_split(x, y, test_size=0.3)

    w, c, v = train_RBF(train_data, train_labels, c_num)
    predictions = test_RBF(test_data, w, c, v)

    plt.scatter(test_data, predictions, label='Predicted', color='red', marker='o')
    plt.plot(x, y, label='data', color='black')

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
