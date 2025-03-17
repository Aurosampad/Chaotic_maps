import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Define the Tent Map function
def tent_map(r, x0, N):
    x = x0
    time_series = []
    for _ in range(N):
        if x < 0.5:
            x = r * x
        else:
            x = r * (1 - x)
        time_series.append(x)
    return np.array(time_series)

# Compute Shannon Entropy
def shannon_entropy(time_series, bins=50):
    hist, _ = np.histogram(time_series, bins=bins, density=True)
    hist = hist[hist > 0]  # Remove zero probabilities
    return entropy(hist, base=2)  # Compute Shannon entropy

# Parameters
r_values = np.linspace(1.0, 2.0, 100)  # Range of r values
entropy_values = []

# Compute entropy for different values of r
for r in r_values:
    time_series = tent_map(r, x0=0.5, N=10000)[500:]  # Remove transient states
    entropy_values.append(shannon_entropy(time_series, bins=50))

# Plot Shannon Entropy vs r
plt.figure(figsize=(8, 5))
plt.plot(r_values, entropy_values, lw=1.5)
plt.xlabel('r')
plt.ylabel('Shannon Entropy')
plt.title('Shannon Entropy of the Tent Map')
plt.grid()
plt.show()
