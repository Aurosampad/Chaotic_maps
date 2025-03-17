import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Logistic Map function
def logistic_map(r, x0, N):
    x = x0
    time_series = []
    for _ in range(N):
        x = r * x * (1 - x)
        time_series.append(x)
    return np.array(time_series)

# Compute Shannon Entropy
def shannon_entropy(time_series, bins=50):
    hist, bin_edges = np.histogram(time_series, bins=bins, density=True)
    hist = hist[hist > 0]  # Remove zero probabilities
    return entropy(hist, base=2)  # Compute Shannon entropy

# Parameters
r_values = np.linspace(2.5, 4.0, 100)  # Range of r values
entropy_values = []

# Compute entropy for different values of r
for r in r_values:
    time_series = logistic_map(r, x0=0.5, N=10000)[500:]  # Remove transient states
    entropy_values.append(shannon_entropy(time_series, bins=50))

# Plot Shannon Entropy vs r
plt.figure(figsize=(8, 5))
plt.plot(r_values, entropy_values, lw=1.5)
plt.xlabel('r')
plt.ylabel('Shannon Entropy')
plt.title('Shannon Entropy of the Logistic Map')
plt.grid()
plt.show()
