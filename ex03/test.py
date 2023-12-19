import numpy as np
from my_linear_regression import MyLinearRegression as MyLR


output_file = "results/ex03/result_ex03.txt"

with open(output_file, "w") as file:
    x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])

    print("--- lr1 ---", file=file)
    print("--- predict_ ---", file=file)
    lr1 = MyLR(np.array([[2], [0.7]]))
    y_hat = lr1.predict_(x)
    print("lr1.predict_(x):\n", lr1.predict_(x), file=file)
    print(
        "expected       :\n",
        np.array(
            [[10.74695094], [17.05055804], [24.08691674], [36.24020866], [42.25621131]]
        ),
        file=file,
    )
    print("--- loss_elem_ ---", file=file)
    print("lr1.loss_elem_(y, y_hat):\n", lr1.loss_elem_(y, y_hat), file=file)
    print(
        "expected       :\n",
        np.array(
            [
                [710.45867381],
                [364.68645485],
                [469.96221651],
                [108.97553412],
                [299.37111101],
            ]
        ),
        file=file,
    )
    print("--- loss_ ---", file=file)
    print("lr1.loss_(y, y_hat):\n", lr1.loss_(y, y_hat), file=file)
    print("expected       :\n", 195.34539903032385, file=file)

    # After fitting and updating thetas, the loss should be smaller
    print("\n--- lr2 ---", file=file)
    lr2 = MyLR(np.array([[1], [1]]), 5e-8, 1500000)
    lr2.fit_(x, y)
    print("--- thetas ---", file=file)

    print("lr2.thetas:\n", lr2.thetas, file=file)
    print("expected       :\n", np.array([[1.40709365], [1.1150909]]), file=file)
    y_hat = lr2.predict_(x)
    print("y_hat:\n", y_hat, file=file)
    print(
        "expected       :\n",
        np.array(
            [[15.3408728], [25.38243697], [36.59126492], [55.95130097], [65.53471499]]
        ),
        file=file,
    )
    print("--- loss_elem_ ---", file=file)

    print("lr2.loss_elem_(y, y_hat):\n", lr2.loss_elem_(y, y_hat), file=file)
    print(
        "expected       :\n",
        np.array(
            [
                [486.66604863],
                [115.88278416],
                [84.16711596],
                [85.96919719],
                [35.71448348],
            ]
        ),
        file=file,
    )
    print("--- loss_ ---", file=file)

    print("lr2.loss_(y, y_hat):\n", lr2.loss_(y, y_hat), file=file)
    print("expected       :\n", 80.83996294128525, file=file)
