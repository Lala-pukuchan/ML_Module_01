import numpy as np
from my_linear_regression import MyLR as MyLR

print("Example 1:")
x = np.array([
        [12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]
        ])
y = np.array([
        [37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]
        ])
lr1 = MyLR(np.array([[2], [0.7]]))
y_hat = lr1.predict_(x)
print(lr1.predict_(x))
print(lr1.loss_(y, y_hat))

print("Example 2:")
lr2 = MyLR(np.array([[1], [1]]), 5e-8, 1500000)
lr2.fit_(x, y)
print(lr2.thetas)
y_hat = lr2.predict_(x)
print(y_hat)
print(lr2.loss_elem_(y, y_hat))
print(lr2.loss_(y, y_hat))
