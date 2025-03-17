import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Generate Sine Map time series
def sine_map(a, x0, N):
    x = x0
    time_series = []
    for _ in range(N):
        x = a * np.sin(np.pi * x)
        time_series.append(x)
    return np.array(time_series)

# Compute Sample Entropy
def sample_entropy(time_series, m=2, r=None):
    N = len(time_series)
    if r is None:
        r = 0.2 * np.std(time_series)  # Set default tolerance as 20% of std dev
    
    def count_matches(seq, m, r):
        patterns = np.array([seq[i:i + m] for i in range(N - m)])
        dists = cdist(patterns, patterns, metric='chebyshev')
        count = np.sum(dists <= r, axis=0) - 1  # Exclude self-matching
        return np.sum(count)
    
    B = count_matches(time_series, m, r)  # Matching patterns of length m
    A = count_matches(time_series, m + 1, r)  # Matching patterns of length m+1
    
    return -np.log(A / B) if A > 0 and B > 0 else np.nan  # Avoid log(0)

# Compute Sample Entropy for different 'a' values
a_values = np.linspace(0.7, 1.0, 100)  # Chaotic behavior for a near 1
entropy_values = []

for a in a_values:
    time_series = sine_map(a, x0=0.5, N=10000)[500:]  # Remove transient states
    entropy_values.append(sample_entropy(time_series))

# Plot Sample Entropy vs a
plt.figure(figsize=(8, 5))
plt.plot(a_values, entropy_values, lw=1.5)
plt.xlabel('a')
plt.ylabel('Sample Entropy')
plt.title('Sample Entropy of the Sine Map')
plt.grid()
plt.show()
