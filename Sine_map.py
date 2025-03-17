import matplotlib.pyplot as plt
import numpy as np
import math
A=np.linspace(0,1,10000)
X=[]
Y=[]
for a in A:
    X.append(a)
    x=np.random.random()
    for n in range(100):
        x=a*math.sin(np.pi*x)
    Y.append(x)
# Plot the bifurcation diagram
plt.figure(figsize=(8, 6))
plt.plot(X, Y, ls='', marker=',', color='black')
plt.xlabel("a")
plt.ylabel("x")
plt.title("Bifurcation Diagram of the Sine Map")
plt.show()