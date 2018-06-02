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
R0 = 50.

# cutoff distance
Rcut = 80.

# RK4 parameters
rtol = 1e-06
atol = 1e-07
maxstep = 100.
ts = 0.
tf = 1E6

# conversion
eV = 27.211396

# numerical differentiation
dtol = 1e-05
dmethod = "stencil"

# H2+ constants
Re = 1.989465870
rovib = np.array(
        [[5.238089122060158e-3, 1.523220332423042e-2, 2.468876822921218e-2,
            3.419259632431514e-2, 4.505482786906121e-2, 5.731330093478369e-2,
            8.571566761008581e-2, 0.191520684613206]]
        )

# root finding for H2 diatomic separation
rmax = 10.
rmin = 0.1

# number of random numbers needed in one trajectory
numRandom = 5
