import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Define the Cascade Combination of Logistic Map and Sine Map
def cascade_map(x, r1, r2):
    x_next = r1 * x * (1 - x)  # Logistic Map
    x_next = r2 * np.sin(np.pi * x_next)  # Sine Map
    return x_next

# Function to compute Sample Entropy (SampEn)
def sample_entropy(time_series, m=2, r=0.2):
    N = len(time_series)
    
    # Create embedding vectors
    def _phi(m):
        patterns = np.array([time_series[i:i + m] for i in range(N - m + 1)])
        distances = cdist(patterns, patterns, metric='chebyshev')  # Max norm
        return np.sum(distances < r) - (N - m + 1)  # Count similar patterns
    
    # Compute probabilities
    A = _phi(m + 1)
    B = _phi(m)
    
    if B == 0:
        return np.inf  # Avoid division by zero
    
    return -np.log(A / B)

# Compute Sample Entropy for different values of r1 (Logistic Map parameter)
def compute_sample_entropy(r1_values, r2, iterations=1000, transient=500):
    sampen_values = []
    
    for r1 in r1_values:
        x = 0.5  # Initial condition
        time_series = []
        
        for i in range(iterations + transient):
            x = cascade_map(x, r1, r2)
            if i >= transient:  # Ignore transient phase
                time_series.append(x)
        
        sampen = sample_entropy(np.array(time_series))
        sampen_values.append(sampen)
    
    return np.array(sampen_values)

# Define parameter range for Logistic Map
r1_values = np.linspace(2.5, 4.0, 200)
r2 = 0.9  # Fixed parameter for Sine Map

# Compute Sample Entropy
sampen_values = compute_sample_entropy(r1_values, r2)

# Plot Sample Entropy
plt.figure(figsize=(10, 5))
plt.plot(r1_values, sampen_values, 'b', lw=1)
plt.xlabel(r'$r_1$ (Logistic Map Parameter)')
plt.ylabel('Sample Entropy')
plt.title('Sample Entropy of Cascade Logistic-Sine Map')
plt.show()
