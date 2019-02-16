import matplotlib.pyplot as plt
import numpy as np


class KMeans:

    def __init__(self, k, tolerance=0.001, max_iter=300):
        self.k = k
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.centroids = {}
        self.groups = {}
        for group in range(self.k):
            self.groups.setdefault(group, [])

    def fit(self, x):
        for cls in range(self.k):
            self.centroids[cls] = x[np.random.choice(len(x))]

        for i in range(self.max_iter):
            for feature in x:
                dist = [np.linalg.norm(np.array(feature) - np.array(self.centroids[c])) for c in self.centroids.keys()]
                group = dist.index(min(dist))
                self.groups[group].append(feature)

            prev_centroids = dict(self.centroids)
            optimized = True

            for group in self.groups.keys():
                self.centroids[group] = np.average(self.groups.get(group), axis=0)

            for centroid in self.centroids.keys():
                if sum(self.centroids[centroid] / (prev_centroids[centroid] * 100)) > self.tolerance:
                    optimized = False

            if optimized:
                break

    def predict(self, x):
        y_pred = []
        for feature in x:
            dist = [np.linalg.norm(np.array(feature) - np.array(self.centroids[c])) for c in self.centroids.keys()]
            y_pred.append(dist.index(min(dist)))
        return y_pred


def show(cls):
    colors = ['g', 'r', 'c', 'b', 'k']

    for centroid in cls.centroids.keys():
        plt.scatter(cls.centroids[centroid][0], cls.centroids[centroid][1],
                    marker='o',
                    color='k',
                    s=100,
                    linewidths=5)

    for group in cls.groups.keys():
        color = colors[group]
        for feature in cls.groups[group]:
            plt.scatter(feature[0], feature[1], marker='x', color=color, s=100, linewidths=5)

    plt.show()


def main():
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8],
                  [1, 0.6], [1, 3], [2, 5], [3, 3], [4, 6]])
    kmeans = KMeans(k=2)
    kmeans.fit(x)
    show(kmeans)
    print(kmeans.predict([[2, 2], [12, 5]]))


if __name__ == '__main__':
    main()

