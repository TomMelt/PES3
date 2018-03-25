import numpy as np
from numpy.linalg import norm
from scipy.constants import codata
import scipy.integrate as odeint
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
# import xarray as xr


# TODO:
# - convert mass to atomic units
# - add Vppp(R1,R2,y) potential
# - take numerical derivative (dVdr) on grid
# - plot Vppp and dVppp in matplotlib
# - currently code uses r, R and cos(gamma)
# - would need to change if gamma is needed


# constants taken from NIST (in u)
# m_H2plus = 2.01533 u
# m_H = 1.00794 u
# m_p = 1.00728 u
# conversion from u to a.u.
m_e = codata.value('electron mass in u')
conversion = 1./m_e
MU_1 = 0.671606*conversion
MU_2 = 0.503805*conversion
Q1 = -1.
Q2 = 1.
Q3 = 1.


def testGridData():
    N = 100
    R1 = np.linspace(0., 10., N)
    R2 = np.linspace(1., 10., N)
    cosGamma = np.linspace(-1., 1., N)
    PES, dVdR1, dVdR2, dVdcosGamma = initialisePESandDerivatives(
            analyticPotential,
            R1,
            R2,
            cosGamma
            )
    print(PES.shape)
    print(dVdR1.shape)
    print(dVdR2.shape)
    print(dVdcosGamma.shape)

    interpdVdR1 = interpolate.RegularGridInterpolator(
            (R1, R2, cosGamma),
            dVdR1
            )
    interpdVdR2 = interpolate.RegularGridInterpolator(
            (R1,R2, cosGamma),
            dVdR2
            )
    interpdVdcosGamma = interpolate.RegularGridInterpolator(
            (R1, R2, cosGamma),
            dVdcosGamma
            )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
#    ax.set_xlabel('R1')
#    ax.set_ylabel('R2')
#    ax.set_zlabel('Z')
    ri, Ri, cosGammai = np.linspace(1.,4.,200), np.linspace(1., 5., 200), np.linspace(-1.,1.,200)
    grid_ri, grid_Ri, grid_cosGammai = np.meshgrid(ri, Ri, cosGammai)
    print(analyticPotential(grid_Ri, grid_ri, grid_cosGammai).shape)
    ax.plot_wireframe(grid_Ri, grid_ri, analyticPotential(grid_Ri, grid_ri, grid_cosGamma), rstride=10, cstride=10)
    quit()
    Z = interp((grid_Ri, grid_ri))
    ax.plot_wireframe(grid_Ri, grid_ri, Z, rstride=3, cstride=3, color='k')
    dZ = np.gradient(potentialGrid,R,r)
    dZdR = np.array(dZ[0])
    dZdr = np.array(dZ[1])
    ax = fig.add_subplot(132, projection='3d')
    ax.plot_wireframe(grid_R, grid_r, dZdR, rstride=10, cstride=10, color='r')
    ax.plot_wireframe(grid_R, grid_r, np.cos(grid_R), rstride=10, cstride=10, color='k')
    ax = fig.add_subplot(133, projection='3d')
    ax.plot_wireframe(grid_R, grid_r, dZdr, rstride=9, cstride=9, color='r')
    ax.plot_wireframe(grid_R, grid_r, -np.sin(grid_r), rstride=10, cstride=10, color='k')
    plt.show()


def analyticPotential(R1, R2, cosGamma):
    alpha = 0.0001
    d1 = np.sqrt(R1*R1-R1*R2*cosGamma+0.25*R2*R2)+alpha
    print(d1)
    d2 = np.sqrt(R1*R1+R1*R2*cosGamma+0.25*R2*R2)+alpha
    V = Q2*Q3/(R2+alpha) - Q1*Q2/d1 - Q1*Q3/d2
    return V


def dVdx(x, y, interpdVdx, interpdVdcosGamma):
    d1 = norm(x)
    d2 = norm(y)
    cosGamma = (x @ y)/(d1*d2)
    position = np.array([d1, d2, cosGamma])
    dVdx = interpdVdx(position)
    dxdx = x/d1
    dVdcosGamma = interpdVdcosGamma(position)
    dcosGammadx = 1./(d2*d1*d1*d1)*(d1*d1*y - (x @ y)*x)
    return dVdx*dxdx+dVdcosGamma*dcosGammadx


def equation_of_motion(
        t,
        coordinates,
        interpdVdR1,
        interpdVdR2,
        interpdVdcosGamma
        ):
    # current jacobi coordinates
    R1, P1, R2, P2 = getJacobiCoordinates(coordinates)

    # first derivatives w.r.t. t
    R1dot = P1/MU_1
    R2dot = P2/MU_2
    P1dot = -dVdx(R1, R2, interpdVdR1, interpdVdcosGamma)
    P2dot = -dVdx(R2, R1, interpdVdR2, interpdVdcosGamma)
    return np.concatenate((R1dot, P1dot, R2dot, P2dot), axis=0)


def getJacobiCoordinates(coordinates):
    R1 = coordinates[:3]
    P1 = coordinates[3:6]
    R2 = coordinates[6:9]
    P2 = coordinates[9:]

    return R1, P1, R2, P2


def getParticleCoordinates(R1, P1, R2, P2):
    r1 = -0.5*R2
    r2 = 0.5*R2
    r3 = R1

    return r1, r2, r3


def initialisePESandDerivatives(potential, R1, R2, cosGamma):
    grid_R1, grid_R2, grid_cosGamma = np.meshgrid(R1, R2, cosGamma)
    grid_PES = potential(grid_R1, grid_R2, cosGamma)
    dV = np.gradient(grid_PES, R1, R2, cosGamma)
    return grid_PES, dV[0], dV[1], dV[2]


def main(args):
    testGridData()
    # initialise PES
    N = 100
    R1 = np.linspace(0.0, 100., N)
    R2 = np.linspace(0.0, 100., N)
    cosGamma = np.linspace(-1., 1., N)
    PES, dVdR1, dVdR2, dVdcosGamma = initialisePESandDerivatives(
            analyticPotential,
            R1,
            R2,
            cosGamma
            )

    # initialise interp for dVdx (x=R1, R2, cosGamma)
    interpdVdR1 = interpolate.RegularGridInterpolator(
            (R1, R2, cosGamma),
            dVdR1
            )
    interpdVdR2 = interpolate.RegularGridInterpolator(
            (R1, R2, cosGamma),
            dVdR2
            )
    interpdVdcosGamma = interpolate.RegularGridInterpolator(
            (R1, R2, cosGamma),
            dVdcosGamma
            )

    # time range
    ts = 0.
    tf = 1000.

    # initial conditions of the scattering particles
    R1 = np.array([0., 0., 5.], dtype=float)
    R2 = np.array([1., 0., 0.], dtype=float)
    P1 = np.array([0., 0., 0.], dtype=float)
    P2 = np.array([0., 0., 0.], dtype=float)
    initialConditions = np.concatenate((R1, P1, R2, P2), axis=0)

    # initialise stepping object
    stepper = odeint.RK45(
            lambda t, y: equation_of_motion(
                t,
                y,
                interpdVdR1,
                interpdVdR2,
                interpdVdcosGamma,
                ),
            ts,
            initialConditions,
            tf,
            rtol=1e-08,
            atol=1e-10
            )

    # array to store trajectories
    data = []

    # propragate Hamilton's eqn's
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    while stepper.t < tf:
        try:
            stepper.step()
            R1, P1, R2, P2 = getJacobiCoordinates(stepper.y)
            r1, r2, r3 = getParticleCoordinates(R1, P1, R2, P2)
            ax.scatter(r1[0], r1[1], r1[2], c='r', marker='.')
            ax.scatter(r2[0], r2[1], r2[2], c='b', marker='.')
            ax.scatter(r3[0], r3[1], r3[2], c='k', marker='.')
            plt.pause(0.1)
            data.append([stepper.t] + stepper.y.tolist())
        except RuntimeError as e:
            print(e)
            break
    plt.ioff()
    data = np.array(data)

#    legend = []
#    for i in range(1, 4):
#        plt.plot(data[:, 0], data[:, i], 'k')
#        plt.plot(dataAnalytic[:, 0], dataAnalytic[:, i], '--r')
#        legend.append(f'r1{i}_n')
#        legend.append(f'r1{i}_a')
#    plt.legend(legend)
#    plt.show()
    return


if __name__ == "__main__":
    main(sys.argv[1:])
