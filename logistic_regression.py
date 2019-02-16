import numpy as np


class LogisticRegression:

    def __init__(self, lr):
        self.lr = lr
        self.weights = None
        pass

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def initialize_weigths(self, shape):
        self.weights = np.random.normal(0, 0.5, shape)

    def update_weights(self, y_true, y_pred, x_train):
        self.weights += self.lr * np.sum(np.dot((y_true - y_pred), x_train))

    def fit(self, x_train, y_train, epochs=10):
        if len(x_train.shape) == 1:
            x_train = x_train.reshape(len(x_train), 1)
        self.initialize_weigths(x_train.shape[1])
        for epoch in range(epochs):
            y_pred = np.round(self.sigmoid(np.dot(self.weights, x_train.T)).reshape(y_train.shape))
            self.update_weights(y_train, y_pred, x_train)

    def predict(self, x):
        if len(x.shape) == 1:
            x = x.reshape(len(x), 1)
        return self.sigmoid(np.dot(self.weights, x.T)).reshape(x.shape[0], )

    def score(self, x, y_true):
        if len(x.shape) == 1:
            x = x.reshape(len(x), 1)
        y_pred = np.round(self.predict(x))
        accuracy = sum([i == j for i, j in zip(y_pred, y_true)]) / len(y_true)
        return 'Accuracy: {}'.format(accuracy)


def main():
    x = np.asarray([x for x in range(50)])
    y = np.asarray([0 if i < 25 else 1 for i in x])
    log = LogisticRegression(0.001)
    log.fit(x, y)
    print(log.score(np.asarray([1, 12]), np.asarray([0, 1])))
    print(log.predict(np.asarray([1, 12])))


if __name__ == '__main__':
    main()
