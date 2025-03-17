import numpy as np
import matplotlib.pyplot as plt

def lyapunov_exponent_sine(r, x0=0.5, N=1000, discard=100):
    x = x0
    sum_log_deriv = 0

    # Discard transient iterations
    for _ in range(discard):
        x = r * np.sin(np.pi * x)

    # Compute Lyapunov exponent
    for _ in range(N):
        x = r * np.sin(np.pi * x)
        sum_log_deriv += np.log(abs(r * np.pi * np.cos(np.pi * x)))

    return sum_log_deriv / N  # Time average

# Compute Lyapunov exponents for different r values
r_values = np.linspace(0.5, 1, 400)  # Sine map is usually defined in r ∈ [0,1]
lyapunov_values = [lyapunov_exponent_sine(r) for r in r_values]

# Plot the Lyapunov exponent as a function of r
plt.figure(figsize=(8, 5))
plt.plot(r_values, lyapunov_values, lw=1)
plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel('r')
plt.ylabel('Lyapunov Exponent')
plt.title('Lyapunov Exponent of the Sine Map')
plt.show()

