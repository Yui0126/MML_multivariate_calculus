#%%
import numpy as np

xdat = [0.4,0.5,0.6,0.7,0.8]
ydat = [0.1,0.25,0.55,0.75,0.85]

xbar = np.sum(xdat)/len(xdat)
ybar = np.sum(ydat)/len(ydat)

print(xbar)
print(ybar)

num = list()
den = list()

for i, y in zip(xdat, ydat):
    n = (i - xbar) * y
    d = (i - xbar) ** 2
    num.append(n)
    den.append(d)

m = sum(num)/sum(den)
c = ybar-m*xbar

print(m)
print(c)

# m = num/den
# c = ybar-m*xbar
# print(m)
# print(c)

