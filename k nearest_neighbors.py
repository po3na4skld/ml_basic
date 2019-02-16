from collections import Counter
import numpy as np


class KNearestNeighbors:

    def __init__(self, k):
        self.k = k
        self.groups = {}
        for group in range(k):
            if group not in self.groups.keys():
                self.groups.setdefault(group, [])

    def fit(self, x, y):
        for feature, group in zip(x, y):
            self.groups[group].append(feature)

    def predict(self, x):
        y_pred = []
        for example in x:
            distances = []
            for group, features in self.groups.items():
                for feature in features:
                    distance = np.linalg.norm(np.array(feature) - np.array(example))
                    distances.append([group, distance])
            votes = [i[0] for i in sorted(distances, key=max)[:self.k]]
            y_pred.append(Counter(votes).most_common(1)[0][0])
        return y_pred

    def score(self, x, y_true):
        y_pred = self.predict(x)
        accuracy = sum([i == j for i, j in zip(y_true, y_pred)]) / len(y_true)
        return 'Accuracy: {}'.format(accuracy)


def main():
    knn = KNearestNeighbors(k=2)
    x = [[5, 5], [4, 4], [10, 12], [11, 15]]
    y = [0, 0, 1, 1]
    x_test = [[1, 1], [13, 20]]
    knn.fit(x, y)
    print(knn.predict(x_test))


if __name__ == '__main__':
    main()
