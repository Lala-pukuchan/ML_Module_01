import os
import sys
import numpy as np

import os
import sys

# Get the absolute path of the current script (fit.py)
current_script_path = os.path.abspath(__file__)

# Get the directory containing the current script
current_dir = os.path.dirname(current_script_path)

# Get the parent directory (which should contain both ex00 and ex02)
parent_dir = os.path.dirname(current_dir)

# Construct the absolute path to the ex00 directory
ex00_dir = os.path.join(parent_dir, 'ex00')

# Add the ex00 directory to sys.path
sys.path.insert(1, ex00_dir)

# Print sys.path for debugging
print(sys.path)

# Import the simple_gradient function from the gradient module in ex00
from gradient import simple_gradient


#def fit_(x, y, theta, alpha, max_iter):
#    """
#    Description:
#            Fits the model to the training dataset contained in x and y.
#    Args:
#            x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
#            y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
#            theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
#            alpha: has to be a float, the learning rate
#            max_iter: has to be an int, the number of iterations done during the gradient descent
#    Returns:
#            new_theta: numpy.ndarray, a vector of dimension 2 * 1.
#            None if there is a matching dimension problem.
#    Raises:
#    """
#    if x.shape[0] != y.shape[0] or theta.shape != (2, 1):
#        return None

#    for _ in range(max_iter):
#        grad = simple_gradient(x, y, theta)
#        if grad is None:
#            return None
#        theta = theta - alpha * grad

#    return theta


#x = np.array(
#    [[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]]
#    )
#y = np.array(
#    [[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]]
#    )
#theta= np.array([1, 1]).reshape((-1, 1))
#theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
#print(theta1)
#print(predict(x, theta1))