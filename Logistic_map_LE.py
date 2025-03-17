import numpy as np
import matplotlib.pyplot as plt

def lyapunov_exponent(r, x0=0.5, N=1000, discard=100):
    x = x0
    sum_log_deriv = 0
    
    # Iterate and discard initial transient
    for _ in range(discard):
        x = r * x * (1 - x)

    # Compute the sum of log |f'(x)|
    for _ in range(N):
        x = r * x * (1 - x)
        sum_log_deriv += np.log(abs(r * (1 - 2*x)))
    
    return sum_log_deriv / N  # Time average

# Compute Lyapunov exponents for different r values
r_values = np.linspace(2.5, 4, 400)
lyapunov_values = [lyapunov_exponent(r) for r in r_values]

# Plot the Lyapunov exponent as a function of r
plt.figure(figsize=(8,5))
plt.plot(r_values, lyapunov_values, lw=1)
plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel('r')
plt.ylabel('Lyapunov Exponent')
plt.title('Lyapunov Exponent of the Logistic Map')
plt.show()
