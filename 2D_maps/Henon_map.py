import matplotlib.pyplot as plt
import numpy as np
import math
a,b=1.4,3
x, y = np.float64(0.01), np.float64(0.01)
X=[]
Y=[]
for _ in range(5000):
    x_new = 1 - a * min(x**2, 1e6) + y
    y_new = b * x
    x, y = x_new, y_new
    X.append(x)
    Y.append(y)
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, s=0.1, color='black')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Hénon Map Attractor")
plt.show()