import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from collections import Counter

# Generate Tent Map time series
def tent_map(mu, x0, N):
    x = x0
    time_series = []
    for _ in range(N):
        x = mu * x if x < 0.5 else mu * (1 - x)
        time_series.append(x)
    return np.array(time_series)

# Compute Permutation Entropy
def permutation_entropy(time_series, m=3, tau=1):
    N = len(time_series)
    patterns = []
    
    # Extract ordinal patterns
    for i in range(N - (m - 1) * tau):
        window = time_series[i:i + m * tau:tau]
        rank_order = tuple(np.argsort(window))  # Get the ordinal pattern
        patterns.append(rank_order)
    
    # Count pattern occurrences
    pattern_counts = Counter(patterns)
    probabilities = np.array(list(pattern_counts.values())) / len(patterns)
    
    # Compute entropy
    return -np.sum(probabilities * np.log2(probabilities))

# Compute Permutation Entropy for different mu values
mu_values = np.linspace(1.0, 2.0, 100)
entropy_values = []

for mu in mu_values:
    time_series = tent_map(mu, x0=0.5, N=10000)[500:]  # Remove transients
    entropy_values.append(permutation_entropy(time_series, m=3))

# Plot Permutation Entropy vs mu
plt.figure(figsize=(8, 5))
plt.plot(mu_values, entropy_values, lw=1.5)
plt.xlabel('mu')
plt.ylabel('Permutation Entropy')
plt.title('Permutation Entropy of the Tent Map')
plt.grid()
plt.show()
