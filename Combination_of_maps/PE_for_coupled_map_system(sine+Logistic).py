import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from itertools import permutations

# Logistic Map
def logistic_map(x, r):
    return r * x * (1 - x)

# Sine Map
def sine_map(y, r):
    return r * np.sin(np.pi * y)

# Coupled Map System
def coupled_map(x, y, r1, r2, epsilon):
    x_next = (1 - epsilon) * logistic_map(x, r1) + epsilon * sine_map(y, r2)
    y_next = (1 - epsilon) * sine_map(y, r2) + epsilon * logistic_map(x, r1)
    return x_next, y_next

# Permutation Entropy Function
def permutation_entropy(time_series, order=3):
    N = len(time_series)
    perm_counts = {perm: 0 for perm in permutations(range(order))}
    
    for i in range(N - order + 1):
        window = time_series[i : i + order]
        perm = tuple(np.argsort(window))
        perm_counts[perm] += 1

    probabilities = np.array(list(perm_counts.values())) / (N - order + 1)
    probabilities = probabilities[probabilities > 0]  # Remove zero probabilities

    return entropy(probabilities, base=2)  # Shannon entropy in base 2

# Generate Coupled Map Time Series
def generate_time_series(r1, r2, epsilon, iterations=1000, transient=500):
    x, y = 0.4, 0.6  # Initial conditions
    time_series = []

    for _ in range(iterations):
        x, y = coupled_map(x, y, r1, r2, epsilon)
        if _ >= transient:
            time_series.append(x)  # Use x-values for entropy calculation

    return np.array(time_series)

# Define Parameters
r1_values = np.linspace(2.5, 4.0, 50)  # Range of r1 values
r2_fixed = 0.9  # Fixed sine map parameter
epsilon = 0.2  # Coupling strength

# Compute Permutation Entropy for Different r1 Values
perm_entropy_values = [
    permutation_entropy(generate_time_series(r1, r2_fixed, epsilon)) for r1 in r1_values
]

# Plot Permutation Entropy
plt.figure(figsize=(10, 6))
plt.plot(r1_values, perm_entropy_values, color='red', lw=1.2, marker='o')
plt.xlabel('r1 (Logistic Map Parameter)')
plt.ylabel('Permutation Entropy')
plt.title('Permutation Entropy of Coupled Logistic-Sine Map System')
plt.grid()
plt.show()
