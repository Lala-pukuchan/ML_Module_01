import numpy as np
from my_linear_regression import MyLinearRegression as MyLR


output_file = "results/ex03/result_ex03.txt"

with open(output_file, "w") as file:

    x = np.array([
        [12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]
        ])
    y = np.array([
        [37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]
        ])

    print("--- lr1 ---", file=file)
    lr1 = MyLR(np.array([[2], [0.7]]))
    y_hat = lr1.predict_(x)
    print("lr1.predict_(x):\n", lr1.predict_(x), file=file)
    print("lr1.loss_elem_(y, y_hat):\n", lr1.loss_elem_(y, y_hat), file=file)
    print("lr1.loss_(y, y_hat):\n", lr1.loss_(y, y_hat), file=file)

    # After fitting and updating thetas, the loss should be smaller
    print("\n--- lr2 ---", file=file)
    lr2 = MyLR(np.array([[1], [1]]), 5e-8, 1500000)
    lr2.fit_(x, y)
    print("lr2.thetas:\n", lr2.thetas, file=file)
    y_hat = lr2.predict_(x)
    print("y_hat:\n", y_hat, file=file)
    print("lr2.loss_elem_(y, y_hat):\n", lr2.loss_elem_(y, y_hat), file=file)
    print("lr2.loss_(y, y_hat):\n", lr2.loss_(y, y_hat), file=file)
