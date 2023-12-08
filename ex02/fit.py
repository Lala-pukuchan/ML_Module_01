import numpy as np


def predict(x, theta):
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if x.size == 0 or theta.size == 0:
        return None
    if theta.shape != (2, 1):
        return None
    y_hat = theta[0] + theta[1] * x
    return y_hat


def gradient(x, y, theta):
    if (
        x.size == 0
        or y.size == 0
        or theta.size == 0
        or x.shape[0] != y.shape[0]
        or theta.shape != (2, 1)
    ):
        return None
    m = x.shape[0]
    x_b = np.hstack((np.ones((m, 1)), x))
    gradient = (1 / m) * x_b.T.dot(x_b.dot(theta) - y)

    return gradient


def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
            Fits the model to the training dataset contained in x and y.
    Args:
            x: has to be a numpy.ndarray, a vector of dimension m * 1:
            (number of training examples, 1).
            y: has to be a numpy.ndarray, a vector of dimension m * 1:
            (number of training examples, 1).
            theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
            alpha: has to be a float, the learning rate
            max_iter: has to be an int, the number of iterations done
            during the gradient descent
    Returns:
            new_theta: numpy.ndarray, a vector of dimension 2 * 1.
            None if there is a matching dimension problem.
    Raises:
    """
    if x.shape[0] != y.shape[0] or theta.shape != (2, 1):
        return None

    for _ in range(max_iter):
        grad = gradient(x, y, theta)
        if grad is None:
            return None
        # grad becomes smaller and smaller when theta is close to minimum
        theta = theta - alpha * grad

    return theta


output_file = "results/ex02/result_ex02.txt"

with open(output_file, "w") as file:

    x = np.array([
        [12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]
        ])
    y = np.array([
        [37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]
        ])
    theta = np.array([1, 1]).reshape((-1, 1))
    theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
    print(theta1, file=file)
    print(predict(x, theta1), file=file)
