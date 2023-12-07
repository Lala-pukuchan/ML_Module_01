import numpy as np


def gradient(x, y, theta):
    """
    Computes the gradient vector from three non-empty numpy arrays,
    without any for loop.
    The three arrays must have compatible shapes.

    Args:
        x: numpy.array, a matrix of shape (m, 1).
        y: numpy.array, a vector of shape (m, 1).
        theta: numpy.array, a vector of shape (2, 1).

    Returns:
        The gradient as a numpy.ndarray, a vector of dimension (2, 1).
        None if x, y, or theta is an empty numpy.ndarray
        or incompatible dimensions.
    """

    # Validate input
    if (
        x.size == 0
        or y.size == 0
        or theta.size == 0
        or x.shape[0] != y.shape[0]
        or theta.shape != (2, 1)
    ):
        return None

    # Reshape x for matrix multiplication
    m = x.shape[0]
    x_b = np.hstack((np.ones((m, 1)), x))

    # Compute the gradient
    gradient = (1 / m) * x_b.T.dot(x_b.dot(theta) - y)

    return gradient


x = np.array(
    [12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]
    ).reshape(
    (-1, 1)
)
y = np.array(
    [37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]
    ).reshape(
    (-1, 1)
)

theta1 = np.array([2, 0.7]).reshape((-1, 1))
print(gradient(x, y, theta1))
theta2 = np.array([1, -0.4]).reshape((-1, 1))
print(gradient(x, y, theta2))
