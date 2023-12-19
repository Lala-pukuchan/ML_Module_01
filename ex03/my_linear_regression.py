import numpy as np


class MyLinearRegression:
    """
    Description:
        My personal linear regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    def gradient(self, x, y):
        if (
            x.size == 0
            or y.size == 0
            or self.thetas.size == 0
            or x.shape[0] != y.shape[0]
            or self.thetas.shape != (2, 1)
        ):
            return None
        m = x.shape[0]
        x_b = np.hstack((np.ones((m, 1)), x))
        gradient = (1 / m) * x_b.T.dot(x_b.dot(self.thetas) - y)

        return gradient

    def fit_(self, x, y):
        if x.shape[0] != y.shape[0] or self.thetas.shape != (2, 1):
            return None

        for _ in range(self.max_iter):
            grad = self.gradient(x, y)
            if grad is None:
                return None
            self.thetas = self.thetas - self.alpha * grad

        return self.thetas

    def predict_(self, x):
        if (not isinstance(x, np.ndarray)
                or not isinstance(self.thetas, np.ndarray)):
            return None
        if x.size == 0 or self.thetas.size == 0:
            return None
        if self.thetas.shape != (2, 1):
            return None
        y_hat = self.thetas[0] + self.thetas[1] * x
        return y_hat

    def loss_elem_(self, y, y_hat):
        if y.shape != y_hat.shape:
            return None
        return (y_hat - y) ** 2

    @staticmethod
    def loss_(y, y_hat):
        if y.shape != y_hat.shape:
            return None
        m = y.shape[0]
        return (1 / (2 * m)) * np.sum((y_hat - y) ** 2)

    @staticmethod
    def mse_(y, y_hat):
        squared_diff = (y_hat - y) ** 2
        if squared_diff is None:
            return None
        return np.mean(squared_diff)
