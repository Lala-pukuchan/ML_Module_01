import numpy as np


def simple_gradient(x, y, theta):
    """
    Computes the gradient vector from three numpy arrays.

    Args:
        x: numpy.array, a vector of shape (m, 1).
        y: numpy.array, a vector of shape (m, 1).
        theta: numpy.array, a vector of shape (2, 1).

    Returns:
        A numpy.array representing the gradient, shape (2, 1),
        or None for invalid input.
    """

    # Check for empty arrays
    if x.size == 0 or y.size == 0 or theta.size == 0:
        return None

    # Check for shape compatibility
    if x.shape != y.shape or theta.shape != (2, 1):
        return None

    # Check for correct data types
    if not (
        isinstance(x, np.ndarray)
        and isinstance(y, np.ndarray)
        and isinstance(theta, np.ndarray)
    ):
        return None

    # Average steepness of the partial derivative curve
    m = x.shape[0]
    gradient = np.zeros((2, 1))

    for i in range(m):
        y_hat = theta[0] + theta[1] * x[i]
        gradient[0] += (y_hat - y[i]) * 1
        gradient[1] += (y_hat - y[i]) * x[i]

    gradient /= m

    return gradient


output_file = "results/ex00/result_ex00.txt"

with open(output_file, "w") as file:

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
    print("gradient with theta1: \n", simple_gradient(x, y, theta1), file=file)

    theta2 = np.array([1, -0.4]).reshape((-1, 1))
    print("gradient with theta2: \n", simple_gradient(x, y, theta2), file=file)
