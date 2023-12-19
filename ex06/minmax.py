import numpy as np


def minmax(x):
    """
    Computes the normalized version of a non-empty numpy.ndarray
    using the min-max standardization.
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

    min_val = np.min(x)
    max_val = np.max(x)

    if max_val == min_val:
        return np.zeros_like(x)

    return ((x - min_val) / (max_val - min_val)).reshape(1, -1)


output_file = "results/ex06/result_ex06.txt"

with open(output_file, "w") as file:
    print("--- minmax X ---", file=file)
    X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
    print(minmax(X), file=file)
    print(
        "expected       :\n",
        np.array(
            [0.58333333, 1.0, 0.33333333, 0.77777778, 0.91666667, 0.66666667, 0.0]
        ),
        file=file,
    )

    print("--- minmax Y ---\n", file=file)
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    print(minmax(Y), file=file)
    print(
        "expected       :\n",
        np.array(
            [0.63636364, 1.0, 0.18181818, 0.72727273, 0.93939394, 0.6969697, 0.0]
        ),
        file=file,
    )
