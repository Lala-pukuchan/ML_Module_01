import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ex03.my_linear_regression import MyLinearRegression as MyLR
from sklearn.metrics import mean_squared_error


output_file = "results/ex04/result_ex04.txt"

with open(output_file, "w") as file:

    data = pd.read_csv("ex04/are_blue_pills_magics.csv")
    Xpill = np.array(data["Micrograms"]).reshape(-1, 1)
    Yscore = np.array(data["Score"]).reshape(-1, 1)

    linear_model1 = MyLR(np.array([[89.0], [-8]]))
    linear_model2 = MyLR(np.array([[89.0], [-6]]))
    Y_model1 = linear_model1.predict_(Xpill)
    Y_model2 = linear_model2.predict_(Xpill)

    print("--- MSE of Y_model1 ---", file=file)
    print("mse of mine        :", MyLR.mse_(Yscore, Y_model1), file=file)
    print("mse of scikit-learn:",
          mean_squared_error(Yscore, Y_model1), file=file)
    print("--- MSE of Y_model2 ---", file=file)
    print("mse of mine        :", MyLR.mse_(Yscore, Y_model2), file=file)
    print("mse of scikit-learn:",
          mean_squared_error(Yscore, Y_model2), file=file)

    plt.figure()
    plt.scatter(Xpill, Yscore, color="blue", label="Actual")
    plt.scatter(Xpill, Y_model1, color="green", label="Predicted")
    plt.plot(Xpill, Y_model1, "g--", label="Predicted")
    plt.xlabel("Quantity of Blue Pills (micrograms)")
    plt.ylabel("Space Driving Score")
    plt.title("Space Driving Score vs Quantity of Blue Pills")
    plt.legend()
    plt.savefig("results/ex04/figure_V1.png")

    n = 6
    theta0_constants = np.linspace(80, 96, n)
    theta1_values = np.linspace(-14, -4, 100)
    plt.figure()
    for theta0 in theta0_constants:
        loss_values = []
        for theta1 in theta1_values:
            linear_model1.thetas = np.array([[theta0], [theta1]])
            Y_model1 = linear_model1.predict_(Xpill)
            loss = linear_model1.loss_(Yscore, Y_model1)
            loss_values.append(loss)

        plt.plot(theta1_values, loss_values, label=f'θ0={theta0}')

    plt.xlabel('θ1')
    plt.ylabel('Loss Function J(θ)')
    plt.title('Evolution of Loss Function J as a function of θ1')
    plt.legend()
    plt.savefig("results/ex04/figure_V2.png")
    plt.close()
