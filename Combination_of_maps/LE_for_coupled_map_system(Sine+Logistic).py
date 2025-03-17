import numpy as np
import matplotlib.pyplot as plt

# Define Logistic Map
def logistic_map(x, r):
    return r * x * (1 - x)

# Define Sine Map
def sine_map(y, r):
    return r * np.sin(np.pi * y)

# Coupled Map System
def coupled_map(x, y, r1, r2, epsilon):
    x_next = (1 - epsilon) * logistic_map(x, r1) + epsilon * sine_map(y, r2)
    y_next = (1 - epsilon) * sine_map(y, r2) + epsilon * logistic_map(x, r1)
    return x_next, y_next

# Function to compute Lyapunov Exponent
def lyapunov_exponent(r1, r2, epsilon, iterations=1000, transient=500):
    x, y = 0.4, 0.6  # Initial conditions
    delta = 1e-8  # Small perturbation for Lyapunov calculation
    le_sum = 0.0

    for i in range(iterations):
        # Evolve the system
        x_next, y_next = coupled_map(x, y, r1, r2, epsilon)
        
        # Perturb the initial condition slightly
        x_perturbed, y_perturbed = coupled_map(x + delta, y, r1, r2, epsilon)

        # Compute distance between original and perturbed trajectories
        d_x = abs(x_perturbed - x_next)
        d_x = max(d_x, 1e-10)  # Avoid log(0) issues

        # Accumulate Lyapunov exponent
        if i >= transient:  # Ignore transient phase
            le_sum += np.log(d_x / delta)

        # Update x and y
        x, y = x_next, y_next

    # Compute Lyapunov exponent
    return le_sum / (iterations - transient)

# Define parameters
r1_values = np.linspace(2.5, 4.0, 500)  # Range of r1 values
r2_fixed = 0.9  # Fixed sine map parameter
epsilon = 0.2  # Coupling strength

# Compute Lyapunov exponents for different r1 values
lyapunov_values = [lyapunov_exponent(r1, r2_fixed, epsilon) for r1 in r1_values]

# Plot Lyapunov Exponent
plt.figure(figsize=(10, 6))
plt.plot(r1_values, lyapunov_values, color='black', lw=0.8)
plt.axhline(0, color='red', linestyle='dashed')  # Zero threshold for chaos
plt.xlabel('r1 (Logistic Map Parameter)')
plt.ylabel('Lyapunov Exponent')
plt.title('Lyapunov Exponent of Coupled Logistic-Sine Map System')
plt.show()
