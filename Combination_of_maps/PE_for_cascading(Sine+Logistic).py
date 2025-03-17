import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

# Define the Cascade Combination of Logistic Map and Sine Map
def cascade_map(x, r1, r2):
    x_next = r1 * x * (1 - x)  # Logistic Map
    x_next = r2 * np.sin(np.pi * x_next)  # Sine Map
    return x_next

# Function to compute Permutation Entropy
def permutation_entropy(time_series, m=3, delay=1):
    N = len(time_series)
    permutations_list = list(permutations(range(m)))  # Generate all possible ordinal patterns
    perm_counts = {p: 0 for p in permutations_list}   # Initialize counts

    # Iterate over the time series to extract patterns
    for i in range(N - (m - 1) * delay):
        pattern = tuple(np.argsort(time_series[i:i + m * delay:delay]))  # Extract pattern
        perm_counts[pattern] += 1

    # Compute probability distribution
    probs = np.array(list(perm_counts.values())) / sum(perm_counts.values())
    probs = probs[probs > 0]  # Avoid log(0)

    # Compute entropy
    return -np.sum(probs * np.log2(probs))

# Compute Permutation Entropy for different values of r1 (Logistic Map parameter)
def compute_permutation_entropy(r1_values, r2, iterations=1000, transient=500, m=3, delay=1):
    permen_values = []
    
    for r1 in r1_values:
        x = 0.5  # Initial condition
        time_series = []

        for i in range(iterations + transient):
            x = cascade_map(x, r1, r2)
            if i >= transient:  # Ignore transient phase
                time_series.append(x)

        # Compute Permutation Entropy
        permen = permutation_entropy(np.array(time_series), m=m, delay=delay)
        permen_values.append(permen)
    
    return np.array(permen_values)

# Define parameter range for Logistic Map
r1_values = np.linspace(2.5, 4.0, 200)
r2 = 0.9  # Fixed parameter for Sine Map

# Compute Permutation Entropy
permen_values = compute_permutation_entropy(r1_values, r2)

# Plot Permutation Entropy
plt.figure(figsize=(10, 5))
plt.plot(r1_values, permen_values, 'r', lw=1)
plt.xlabel(r'$r_1$ (Logistic Map Parameter)')
plt.ylabel('Permutation Entropy')
plt.title('Permutation Entropy of Cascade Logistic-Sine Map')
plt.show()
