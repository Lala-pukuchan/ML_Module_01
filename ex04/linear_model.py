import numpy as np
import pandas as pd
from ex03.my_linear_regression import MyLinearRegression as MyLR
from sklearn.metrics import mean_squared_error

output_file = "result/result_ex04.txt"
with open(output_file, "w") as file:
    data = pd.read_csv("ex04/are_blue_pills_magics.csv")
    Xpill = np.array(data["Micrograms"]).reshape(-1, 1)
    Yscore = np.array(data["Score"]).reshape(-1, 1)

    linear_model1 = MyLR(np.array([[89.0], [-8]]))
    linear_model2 = MyLR(np.array([[89.0], [-6]]))
    Y_model1 = linear_model1.predict_(Xpill)
    Y_model2 = linear_model2.predict_(Xpill)

    print("--- test1 ---", file=file)
    print("mse of mine:", MyLR.mse_(Yscore, Y_model1), file=file)
    print("mse of scikit-learn:", mean_squared_error(Yscore, Y_model1), file=file)
    print("--- test2 ---", file=file)
    print("mse of mine:", MyLR.mse_(Yscore, Y_model2), file=file)
    print("mse of scikit-learn:", mean_squared_error(Yscore, Y_model2), file=file)
