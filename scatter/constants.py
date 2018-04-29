import numpy as np

# mass constants
# currently set for Hydrogen (Triatomic)
m1 = 1.00794/0.00054858
m2 = 1.00794/0.00054858
mu = m1*m2/(m1+m2)
m3 = 1.00794/0.00054858
Mt = m1+m2+m3
MU = m3*(m1+m2)/Mt

# initial separation of scattering particle R(t=0)
R0 = 5.

# scattering range max
bmax = 2.

# RK4 parameters
rtol = 1e-06
atol = 1e-07
maxstep = 100.
ts = 0.
tf = 20000.

# conversion
eV = 27.211396

# elastic cutoff
cutoff = 1e-4

# H2 rovibrational constants
G = np.array([0.545546, -0.01497822, 8.9783E-5])/eV
F = np.array([
    [0.00754283, -3.74130E-4, 3.540E-6],
    [-5.8072E-6, 2.115E-7, -3.82E-9],
    [6.46E-19, 0., 0.]])/eV

# root finding for H2 diatomic separation
rmax = 4.
rmin = 0.1
