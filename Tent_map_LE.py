import numpy as np
import matplotlib.pyplot as plt

def lyapunov_exponent_tent(r, x0=0.5, N=1000, discard=100):
    x = x0
    sum_log_deriv = 0

    # Discard transient iterations
    for _ in range(discard):
        x = r * x if x < 0.5 else r * (1 - x)

    # Compute Lyapunov exponent
    for _ in range(N):
        x = r * x if x < 0.5 else r * (1 - x)
        sum_log_deriv += np.log(abs(r))  # Since derivative is r or -r

    return sum_log_deriv / N  # Time average

# Compute Lyapunov exponents for different r values
r_values = np.linspace(0.5, 2, 400)  # r typically varies in [0,2] for the Tent Map
lyapunov_values = [lyapunov_exponent_tent(r) for r in r_values]

# Plot the Lyapunov exponent as a function of r
plt.figure(figsize=(8, 5))
plt.plot(r_values, lyapunov_values, lw=1)
plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel('r')
plt.ylabel('Lyapunov Exponent')
plt.title('Lyapunov Exponent of the Tent Map')
plt.show()
