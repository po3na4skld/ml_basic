import matplotlib.pyplot as plt
import random


class LinearRegression:

    def __init__(self, data):
        self.x = data[0]
        self.y = data[1]
        self._a = self.a
        self._b = self.b
        self._line = self.line

    @property
    def a(self):
        self._a = sum(self.y) * sum(x**2 for x in self.x) - sum(self.x) * sum(x*y for x, y in zip(self.x, self.y))
        self._a /= len(self.x) * sum(x**2 for x in self.x) - sum(self.x)**2
        return self._a

    @property
    def b(self):
        self._b = len(self.x) * sum(x*y for x, y in zip(self.x, self.y)) - sum(self.x) * sum(self.y)
        self._b /= len(self.x) * sum(x**2 for x in self.x) - sum(self.x)**2
        return self._b

    @property
    def line(self):
        self._line = [self.a + self.b * x for x in self.x]
        return self._line

    def plot_line(self):
        plt.style.use('fivethirtyeight')
        plt.scatter(self.x, self.y, c='red')
        plt.plot(self.x, self.line, c='green')
        plt.show()


def create_dataset(lenght, variance, step=2):
    x, y = [x for x in range(lenght)], []
    value = 1
    for i in range(lenght):
        y.append(value + random.randrange(-variance, variance))
        value += step
    return x, y


def main():
    data = create_dataset(20, 2)
    linear_regression = LinearRegression(data)
    linear_regression.plot_line()


if __name__ == '__main__':
    main()
