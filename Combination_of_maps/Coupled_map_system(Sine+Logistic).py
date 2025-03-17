import numpy as np
import matplotlib.pyplot as plt

# Define the Logistic and Sine maps
def logistic_map(x, r):
    return r * x * (1 - x)

def sine_map(y, r):
    return r * np.sin(np.pi * y)

# Coupled Map System
def coupled_map_bifurcation(r1_vals, r2, epsilon, iterations=1000, transient=500):
    x_init, y_init = 0.4, 0.6  # Initial conditions
    x_bif, r_bif = [], []

    for r1 in r1_vals:
        x, y = x_init, y_init  # Reset initial conditions for each r1

        for i in range(iterations):
            x_next = (1 - epsilon) * logistic_map(x, r1) + epsilon * sine_map(y, r2)
            y_next = (1 - epsilon) * sine_map(y, r2) + epsilon * logistic_map(x, r1)
            x, y = x_next, y_next

            if i >= transient:  # Ignore transient phase
                x_bif.append(x)
                r_bif.append(r1)

    return r_bif, x_bif

# Define parameters
r1_values = np.linspace(2.5, 4.0, 500)  # Range of r1 values for bifurcation
r2_fixed = 0.9  # Fixed sine map parameter
epsilon = 0.2  # Coupling strength

# Compute bifurcation
r_vals, x_vals = coupled_map_bifurcation(r1_values, r2_fixed, epsilon)

# Plot bifurcation diagram
plt.figure(figsize=(10, 6))
plt.scatter(r_vals, x_vals, s=0.1, color='black', alpha=0.7)
plt.xlabel('r1 (Logistic Map Parameter)')
plt.ylabel('Asymptotic x Values')
plt.title('Bifurcation Diagram of Coupled Logistic-Sine Map')
plt.show()

