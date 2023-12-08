import numpy as np
from ex03.my_linear_regression import MyLinearRegression

output_file = "output.txt"

with open(output_file, "w") as file:
    print("Example 1:", file=file)
    
    x = np.array([
            [12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]
            ])
    y = np.array([
            [37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]
            ])
    lr1 = MyLinearRegression(np.array([[2], [0.7]]))
    y_hat = lr1.predict_(x)

    # Printing to the file
    np.savetxt(file, lr1.predict_(x), fmt='%s', header='Predicted Y:')
    print(f"Loss: {lr1.loss_(y, y_hat)}", file=file)
