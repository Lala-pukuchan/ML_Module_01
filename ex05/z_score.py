import numpy as np


def zscore(x):
    """
    Computes the normalized version of a non-empty numpy.ndarray
    using the z-score standardization.
    Args:
        x: has to be an numpy.ndarray, a vector.
    Returns:
        x' as a numpy.ndarray.
        None if x is an empty numpy.ndarray or not a numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or x.size == 0:
        return None

    return ((x - np.mean(x)) / np.std(x)).reshape((1, -1))


output_file = "results/ex05/result_ex05.txt"

with open(output_file, "w") as file:

    X = np.array([0, 15, -9, 7, 12, 3, -21])
    print(zscore(X), file=file)
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    print(zscore(Y), file=file)
