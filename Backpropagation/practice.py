
#%%

# below is for practice
import numpy as np

n1 = 7
n2 = 6

x = np.random.randn(n2, n1)
x = x/2
print(x)

#%%

t1 = np.floor(np.random.randn(6,1))
t2 = np.floor(np.random.randn(1,3))

t3 = t1*t2
t4 = t1@t2

print(t1)
print(t2)
print(t3)
print(t4)

#%%

import numpy as np
# import matplotlib.pyplot as plt


sigma = lambda z : 1 / (1 + np.exp(-z)) #logistic function
d_sigma = lambda z : np.cosh(z/2)**(-2) / 4


def reset_network (n1 = 6, n2 = 7, random=np.random) :
    global W1, W2, W3, b1, b2, b3
    W1 = random.randn(n1, 1) / 2
    W2 = random.randn(n2, n1) / 2
    W3 = random.randn(2, n2) / 2
    b1 = random.randn(n1, 1) / 2
    b2 = random.randn(n2, 1) / 2
    b3 = random.randn(2, 1) / 2

print(W3)




#%%

def network_function(a0) :
    z1 = W1 @ a0 + b1
    a1 = sigma(z1)
    z2 = W2 @ a1 + b2
    a2 = sigma(z2)
    z3 = W3 @ a2 + b3
    a3 = sigma(z3)
    return a1, z1, a1, z2, a2, z3, a3

def cost(x, y) :
    return np.linalg.norm(network_function(x)[-1] - y)**2 / x.size

def J_W3 (x, y) :
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    J = 2 * (a3 - y)
    J = J * d_sigma(z3)
    J = J @ a2.T / x.size
    return J

def J_b3 (x, y) :
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    J = 2 * (a3 - y)
    J = J * d_sigma(z3)
    J = np.sum(J, axis=1, keepdims=True) / x.size
    return J

a = np.random.randn(1,3)
b = np.random.randn(1,3)

c = cost(a, b)
print(c)
w = J_W3(a, b)
print(w)
b = J_b3(a, b)
print(b)



#%%

def J_W2 (x, y) :
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)    
    J = 2 * (a3 - y)
    J = J * d_sigma(z3)
    J = (J.T @ W3).T
    J = J * d_sigma(z2)
    J = J @ a1.T / x.size
    return J

def J_b2 (x, y) :
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    J = 2 * (a3 - y)
    J = J * d_sigma(z3)
    J = (J.T @ W3).T
    J = J * d_sigma(z2)
    J = np.sum(J, axis=1, keepdims=True) / x.size
    return J


def J_W1 (x, y) :
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    J = 2*(a3-y)
    J = J*d_sigma(z3)
    J = (J.T @ W3).T
    J = J*d_sigma(z2)
    J = (J.T @ W2).T
    J = J*d_sigma(z1)
    J = J @ a0.T / x.size
    return J

def J_b1 (x, y) :
    a0, z1, a1, z2, a2, z3, a3 = network_function(x)
    J = 2*(a3-y)
    J = J*d_sigma(z3)
    J = (J.T @ W3).T
    J = J*d_sigma(z2)
    J = (J.T @ W2).T
    J = J*d_sigma(z1)
    J = np.sum(J, axis=1, keepdims=True) / x.size
    return J

