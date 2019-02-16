import numpy as np


class PCA:

    def __init__(self, data):
        self.data = data.astype('float64')

    def transform(self):
        m = np.mean(self.data.T, axis=1)
        self.data -= m
        v = np.cov(self.data.T)
        values, vectors = np.linalg.eig(v)
        p = vectors.T.dot(self.data.T)
        return p.T


def main():
    a = np.asarray([[1, 2],
                    [3, 4],
                    [5, 6]])
    pca = PCA(a)
    print('Before transformation:\n', pca.data)
    print('After transformation:\n', pca.transform())


if __name__ == '__main__':
    main()
