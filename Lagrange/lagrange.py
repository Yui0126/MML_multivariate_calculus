#%%

from scipy import optimize
import numpy as np

# Define the objective function f(x, y)
def f(x, y):
    return np.exp(-(2*x**2 + y**2 - x*y) / 2)

# Define the constraint function g(x, y)
def g(x, y):
    return x**2 + 3*(y+1)**2 - 1

# Define the partial derivatives of f and g
def dfdx(x, y):
    return -x * np.exp(-(2*x**2 + y**2 - x*y)/2)

def dfdy(x, y):
    return -y * np.exp(-(2*x**2 + y**2 - x*y)/2)

def dgdx(x, y):
    return 2*x

def dgdy(x, y):
    return 6*(y+1)

# Define the Lagrangian function and its derivatives
def DL(xyλ):
    [x, y, λ] = xyλ
    return np.array([
            dfdx(x, y) - λ * dgdx(x, y),
            dfdy(x, y) - λ * dgdy(x, y),
            -g(x, y)
        ])

# Try different initial guesses for finding other solutions
initial_guesses = [(-1, -1, 0), (1, 1, 0), (-1, 1, 0), (1, -1, 0)]

for i, (x0, y0, λ0) in enumerate(initial_guesses, 1):
    # Find the root using the current initial guess
    x, y, λ = optimize.root(DL, [x0, y0, λ0]).x
    
    # Print the results
    print(f"Solution {i}:")
    print("x =", x)
    print("y =", round(y, 3))  # Round y to two decimal places
    print("λ =", λ)
    print("f(x, y) =", f(x, y))
    print()
