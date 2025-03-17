import numpy as np
import matplotlib.pyplot as plt

# Define the cascade combination of Logistic and Sine Maps
def cascade_map(x, r1, r2):
    x = r1 * x * (1 - x)  # Logistic Map
    x = r2 * np.sin(np.pi * x)  # Sine Map
    return x

# Parameters
r1_values = np.linspace(2.5, 4.0, 500)  # Logistic Map parameter
r2 = 0.9  # Fixed Sine Map parameter
iterations = 1000
last = 200  # Last points to plot (after transient)

# Prepare the figure
plt.figure(figsize=(10, 6))
for r1 in r1_values:
    x = 0.5  # Initial condition
    for _ in range(iterations):
        x = cascade_map(x, r1, r2)  # Apply cascade map
        if _ >= (iterations - last):  # Plot last points after transient
            plt.plot(r1, x, ',k', alpha=0.25)

plt.xlabel(r'$r_1$ (Logistic Map Parameter)')
plt.ylabel('X values')
plt.title('Bifurcation Diagram of Cascade Logistic-Sine Map')
plt.show()
