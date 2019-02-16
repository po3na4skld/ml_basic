import matplotlib.pyplot as plt
import numpy as np


class SupportVectorMachine:

    def __init__(self, lr, epochs):
        self.weights = None
        self.lr = lr
        self.epochs = epochs
        self.errors = []

    def initialize_weights(self, x):
        self.weights = np.zeros(len(x[0]))

    def fit(self, x, y):
        self.initialize_weights(x)
        for epoch in range(1, self.epochs):
            error = 0
            for i, _ in enumerate(x):
                if (y[i] * np.dot(x[i], self.weights)) < 1:
                    self.weights += self.lr * (x[i] * y[i] + (-2 * (1/epoch) * self.weights))
                    error = 1
                else:
                    self.weights += self.lr * (-2 * (1/epoch) * self.weights)
            self.errors.append(error)

    def show(self, x):
        for d, sample in enumerate(x):
            if d < 2:
                plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
            else:
                plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

        x2 = [self.weights[0], self.weights[1], -self.weights[1], self.weights[0]]
        x3 = [self.weights[0], self.weights[1], self.weights[1], -self.weights[0]]
        hyperplane = np.array([x2, x3])
        X, Y, U, V = zip(*hyperplane)
        ax = plt.gca()
        ax.quiver(X, Y, U, V, scale=1, color='blue')
        plt.show()


def main():
    x = np.array([[-2, 4, -1],
                 [4, 1, -1],
                 [1, 6, -1],
                 [2, 4, -1],
                 [6, 2, -1]])
    y = np.array([-1, -1, 1, 1, 1])
    svm = SupportVectorMachine(1, 10000)
    svm.fit(x, y)
    svm.show(x)


if __name__ == '__main__':
    main()
