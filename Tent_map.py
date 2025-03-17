import matplotlib.pyplot as plt
import numpy as np

# Define parameter space for u
U = np.linspace(0, 1, 1000)  # Values of 'u' from 0 to 1
iterations = 500  # Total iterations
last = 100  # Number of points to plot (after transient)

X = []
Y = []

for u in U:
    x = np.random.random()  # Initial random value
    # Iterate the Tent Map function
    for _ in range(iterations):
        if x < u:
            x = x / u
        else:
            x = (1 - x) / (1 - u)
        
        # Store only the last 'last' iterations for bifurcation structure
        if _ >= (iterations - last):
            X.append(u)
            Y.append(x)

# Plot the bifurcation diagram
plt.figure(figsize=(8, 6))
plt.plot(X, Y, ls='', marker=',', color='black')
plt.xlabel("u")
plt.ylabel("x")
plt.title("Bifurcation Diagram of the Tent Map")
plt.show()

        