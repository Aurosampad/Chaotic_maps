import matplotlib.pyplot as plt
import numpy as np
R=np.linspace(3.57,4,10000)
X=[]
Y=[]
for r in R:
    X.append(r)
    x=np.random.random()
    for n in range(100):
        x=r*x*(1-x)
    Y.append(x)
plt.figure(figsize=(8, 6))
plt.plot(X, Y, ls='', marker=',', color='black')
plt.xlabel("r")
plt.ylabel("x")
plt.title("Bifurcation Diagram of the Logistic Map")
plt.show()