import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Generate Logistic Map time series
def logistic_map(r, x0, N):
    x = x0
    time_series = []
    for _ in range(N):
        x = r * x * (1 - x)
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

# Compute Sample Entropy for different r values
r_values = np.linspace(2.5, 4.0, 100)
entropy_values = []

for r in r_values:
    time_series = logistic_map(r, x0=0.5, N=10000)[500:]  # Remove transient states
    entropy_values.append(sample_entropy(time_series))

# Plot Sample Entropy vs r
plt.figure(figsize=(8, 5))
plt.plot(r_values, entropy_values, lw=1.5)
plt.xlabel('r')
plt.ylabel('Sample Entropy')
plt.title('Sample Entropy of the Logistic Map')
plt.grid()
plt.show()
