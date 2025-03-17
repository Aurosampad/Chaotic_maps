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

# Sample Entropy Function
def sample_entropy(time_series, m=2, r=0.2):
    N = len(time_series)
    r *= np.std(time_series)  # Scale r based on time series std deviation
    
    def _phi(m):
        count = 0
        for i in range(N - m):
            template = time_series[i:i + m]
            for j in range(i + 1, N - m):
                comparison = time_series[j:j + m]
                if np.linalg.norm(template - comparison, ord=np.inf) < r:
                    count += 1
        return count / (N - m)

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    
    return -np.log(phi_m1 / phi_m) if phi_m1 > 0 else np.inf

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

# Compute Sample Entropy for Different r1 Values
sample_entropy_values = [
    sample_entropy(generate_time_series(r1, r2_fixed, epsilon)) for r1 in r1_values
]

# Plot Sample Entropy
plt.figure(figsize=(10, 6))
plt.plot(r1_values, sample_entropy_values, color='blue', lw=1.2, marker='o')
plt.xlabel('r1 (Logistic Map Parameter)')
plt.ylabel('Sample Entropy')
plt.title('Sample Entropy of Coupled Logistic-Sine Map System')
plt.grid()
plt.show()
