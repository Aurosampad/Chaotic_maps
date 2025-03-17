import numpy as np
import matplotlib.pyplot as plt

# Define the Cascade Combination of Logistic Map and Sine Map
def cascade_map(x, r1, r2):
    x_next = r1 * x * (1 - x)  # Logistic Map
    x_next = r2 * np.sin(np.pi * x_next)  # Sine Map
    return x_next

# Compute Lyapunov Exponent
def lyapunov_exponent(r1_values, r2, iterations=1000, transient=500):
    lyapunov_exp = []
    for r1 in r1_values:
        x = 0.5  # Initial condition
        sum_lyap = 0
        
        for i in range(iterations + transient):
            x_next = cascade_map(x, r1, r2)
            derivative = abs(r1 * (1 - 2*x) * r2 * np.pi * np.cos(np.pi * x_next))
            
            if i >= transient:  # Ignore transient phase
                if derivative > 0:  # Avoid log(0) errors
                    sum_lyap += np.log(derivative)
            
            x = x_next  # Update state
            
        lyapunov_exp.append(sum_lyap / iterations)
    
    return np.array(lyapunov_exp)

# Define range for r1 (Logistic Map parameter)
r1_values = np.linspace(2.5, 4.0, 500)
r2 = 0.9  # Fixed Sine Map parameter

# Compute Lyapunov Exponent
lyap_exp = lyapunov_exponent(r1_values, r2)

# Plot Lyapunov Exponent
plt.figure(figsize=(10, 5))
plt.plot(r1_values, lyap_exp, 'b', lw=1)
plt.axhline(y=0, color='k', linestyle='--')  # Zero line to distinguish chaos
plt.xlabel(r'$r_1$ (Logistic Map Parameter)')
plt.ylabel('Lyapunov Exponent')
plt.title('Lyapunov Exponent of Cascade Logistic-Sine Map')
plt.show()
