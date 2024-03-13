import numpy as np



# This is the Gaussian function.
def f (x,mu,sig) :
    return np.exp(-(x-mu)**2/(2*sig**2)) / np.sqrt(2*np.pi) / sig

# Next up, the derivative with respect to μ.
# If you wish, you may want to express this as f(x, mu, sig) multiplied by chain rule terms.
# === COMPLETE THIS FUNCTION ===
def dfdmu (x,mu,sig) :
    return f(x, mu, sig) * ((x-mu)/sig**2)

# Finally in this cell, the derivative with respect to σ.
# === COMPLETE THIS FUNCTION ===

def df_dsigma(x, mu, sig):
    exponent = -(x - mu)**2 / (2 * sig**2)
    factor = (x - mu)**2 / (sig**3)
    result = -1 / sig**2 * np.exp(exponent) / (np.sqrt(2 * np.pi)) + factor * np.exp(exponent) / (np.sqrt(2 * np.pi) * sig)
    return result
