import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Generate Tent Map time series
def tent_map(mu, x0, N):
    x = x0
    time_series = []
    for _ in range(N):
        x = mu * x if x < 0.5 else mu * (1 - x)
        time_series.append(x)
    return np.array(time_series)

# Compute Sample Entropy
def sample_entropy(time_series, m=2, r=None):
    N = len(time_series)
    if r is None:
        r = 0.2 * np.std(time_series)  # Default tolerance as 20% of std dev
    
    def count_matches(seq, m, r):
        patterns = np.array([seq[i:i + m] for i in range(N - m)])
        dists = cdist(patterns, patterns, metric='chebyshev')
        count = np.sum(dists <= r, axis=0) - 1  # Exclude self-matching
        return np.sum(count)
    
    B = count_matches(time_series, m, r)  # Matching patterns of length m
    A = count_matches(time_series, m + 1, r)  # Matching patterns of length m+1
    
    return -np.log(A / B) if A > 0 and B > 0 else np.nan  # Avoid log(0)

# Compute Sample Entropy for different 'mu' values
mu_values = np.linspace(1.0, 2.0, 100)  # Chaotic behavior for mu > 1.4
entropy_values = []

for mu in mu_values:
    time_series = tent_map(mu, x0=0.5, N=10000)[500:]  # Remove transient states
    entropy_values.append(sample_entropy(time_series))

# Plot Sample Entropy vs mu
plt.figure(figsize=(8, 5))
plt.plot(mu_values, entropy_values, lw=1.5)
plt.xlabel('mu')
plt.ylabel('Sample Entropy')
plt.title('Sample Entropy of the Tent Map')
plt.grid()
plt.show()
