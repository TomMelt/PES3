import numpy as np
from numpy import sin, cos, arccos, tan
# from numpy.linalg import norm
# from scipy.constants import codata
from scipy.special import legendre
import scipy.integrate as odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
# import xarray as xr


# TODO:

# NOTE:
# - when dr is set equal to 0. instead of pr/mu
#   the Hamiltonian is no longer conserved!!

# constants taken from NIST (in u)
# m_H2plus = 2.01533 u
# m_H = 1.00794 u
# m_p = 1.00728 u
# conversion from u to a.u.
# m_e = codata.value('electron mass in u')
# conversion = 1./m_e
# MU_1 = 0.671606*conversion
# MU_2 = 0.503805*conversion
m1 = 1.
m2 = 1.
mu = m1*m2/(m1+m2)
M = 1.


def potential(r, R, gamma):
    alpha = 2.027
    beta = 0.375
    C = 17.283
    p2 = legendre(2)
    V = C*np.exp(-alpha*R)*(1.+beta*p2(np.cos(gamma)))
    return V


def dVdR(r, R, gamma):
    alpha = 2.027
    return -alpha*potential(r, R, gamma)


def dVdgamma(r, R, gamma):
    alpha = 2.027
    beta = 0.375
    C = 17.283
    return -3.*beta*C*np.exp(-alpha*R)*np.cos(gamma)*np.sin(gamma)


def Hamiltonian(r, t, p, R, T, P, pr, pt, pp, pR, pT, pP):
    KE11 = pr*pr/(2.*mu) 
    if p == 0.:
        KE12 = 0.
    else:
        KE12 = pt*pt/(2.*mu*r*r*sin(p)*sin(p))
    KE13 = pp*pp/(2.*mu*r*r)

    KE21 = pR*pR/(2.*M)
    if P == 0.:
        KE22 = 0.
    else:
        KE22 = pT*pT/(2.*M*R*R*sin(P)*sin(P))
    KE23 = pP*pP/(2.*M*R*R)
    # gamma = arccos(sin(p)*sin(P)*cos(t-T) + cos(p)*cos(P))
    # return KE1 + KE2 + potential(r, R, gamma)
    KE1 = KE11 + KE12 + KE13
    KE2 = KE21 + KE22 + KE23

    return KE1 + KE2


def derivative(func, x, h=1e-8):
    return 0.5*(func(x+h)-func(x-h))/h


# def numericDerivatives(r, R):
#
#    #rmag = norm(r)
#    #Rmag = norm(R)
#    #cosGamma = (r @ R)/(rmag * Rmag)
#    #
#    #dVdr = derivative(lambda x: potential(x, Rmag, cosGamma), rmag)
#    #dVdR = derivative(lambda x: potential(rmag, x, cosGamma), Rmag)
#    #dVdcosGamma = derivative(lambda x: potential(rmag, Rmag, x), cosGamma)
#    #dcosGammadr = (rmag/Rmag*R - cosGamma*r)/(rmag*rmag)
#    #dcosGammadR = (Rmag/rmag*r - cosGamma*R)/(Rmag*Rmag)
#    #pdot = -dVdr*r/rmag - dVdcosGamma*dcosGammadr
#    #Pdot = -dVdR*R/Rmag - dVdcosGamma*dcosGammadR
#
#    return pdot, Pdot

def analyticDerivatives(
        r, t, p, R, T, P,
        pr, pt, pp, pR, pT, pP
        ):
    # calculate and return the analytic derivatives

    dr = 0.
    if p == 0.:
        dt = 0.
    else:
        dt = pt/(mu*r*r*sin(p)*sin(p))
    dp = pp/(mu*r*r)

    dR = pR/M
    if P == 0.:
        dT = 0.
    else:
        dT = pT/(M*R*R*sin(P)*sin(P))
    dP = pP/(M*R*R)

    dpr = pt/r*dt + pp/r*dp
    dpt = 0.
    if p == 0.:
        dpp = 0.
    else:
        dpp = pt/(2.*tan(p))*dt

    dpR = pT/R*dT + pP/R*dP
    dpT = 0.
    if P == 0.:
        dpP = 0.
    else:
        dpP = pT/(2.*tan(P))*dT

    return dr, dt, dp, dR, dT, dP, dpr, dpt, dpp, dpR, dpT, dpP


def equation_of_motion(t, coordinates, numeric=True):
    # current jacobi coordinates
    r, t, p, R, T, P, pr, pt, pp, pR, pT, pP = getPropCoords(coordinates)

    if numeric:
        return
    else:
        derivs = list(
                analyticDerivatives(
                    r, t, p, R, T, P,
                    pr, pt, pp, pR, pT, pP
                    )
                )
        return np.array(derivs)


def getPropCoords(coordinates):
    # Get propagation coordinates (phase-space configuration)

    r = coordinates[0]
    t = coordinates[1]
    p = coordinates[2]
    R = coordinates[3]
    T = coordinates[4]
    P = coordinates[5]
    pr = coordinates[6]
    pt = coordinates[7]
    pp = coordinates[8]
    pR = coordinates[9]
    pT = coordinates[10]
    pP = coordinates[11]

    return r, t, p, R, T, P, pr, pt, pp, pR, pT, pP


def getParticleCoords(
        r, t, p, R, T, P,
        pr, pt, pp, pR, pT, pP
        ):
    # Get cartesian position of each particle
    r1 = m2/(m1+m2)*r*np.array([cos(t)*sin(p), sin(t)*sin(p), cos(p)])
    r2 = -m1/(m1+m2)*r*np.array([cos(t)*sin(p), sin(t)*sin(p), cos(p)])
    r3 = R*np.array([cos(T)*sin(P), sin(T)*sin(P), cos(P)])

    return r1, r2, r3


def main(args):
    #    testDeriv()

    # time range
    ts = 0.
    tf = 1000.

    # initial conditions of the scattering particles
    # r, theta, R, phi, pr, ptheta, pR, pphi
    r_ini = np.array([2., 0., np.pi/2.])
    R_ini = np.array([5., 0., 0.])
    p_ini = np.array([0., 0.1, 0.])
    P_ini = np.array([-0.1, 0., 0.])

    r_ini = np.array([2., 0., np.pi/2.])
    R_ini = np.array([1./sin(np.pi/4.), 0., np.pi/4.])
    p_ini = np.array([0., 0., 0.])
    P_ini = M*np.array([-cos(R_ini[2]), 0., R_ini[0]*sin(R_ini[2])])

    r_ini = np.array([2., 0., np.pi/2.])
    R_ini = np.array([5., 0., 0.])
    p_ini = np.array([0., 0.1, 0.1])
    P_ini = np.array([-0.1, 0., 0.])

    initialConditions = np.hstack((r_ini, R_ini, p_ini, P_ini))

    # initialise stepping object
    stepper = odeint.RK45(
            lambda t, y: equation_of_motion(t, y, numeric=False),
            ts,
            initialConditions,
            tf,
            max_step=1e1,
            rtol=1e-08,
            atol=1e-10
            )

    # array to store trajectories
    data = []

    # propragate Hamilton's eqn's
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    while stepper.t < tf:
        try:
            stepper.step()
            print(stepper.step_size)
            r, t, p, R, T, P, pr, pt, pp, pR, pT, pP = getPropCoords(stepper.y)
            r1, r2, r3 = getParticleCoords(
                    r, t, p, R, T, P,
                    pr, pt, pp, pR, pT, pP
                    )
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


def testDeriv():
    N = 100
    r = np.linspace(0, 10, N)
    R = np.linspace(0.5, 10, N)
    cosGamma = np.linspace(-1, 1, N)
    grid_r, grid_R, grid_cosGamma = np.meshgrid(r, R, cosGamma)
    # grid_Pot = potential(grid_r, grid_R, cosGamma)
    fig = plt.figure()
    stride = 1
    strideAna = 5

    alpha = 2.027
    beta = 0.375
    C = 17.283

    dVdr = derivative(lambda x: potential(x, grid_R, grid_cosGamma), grid_r)
    dVdrAnalytic = np.zeros(dVdr.shape)
    ax = fig.add_subplot(131, projection='3d')
    ax.plot_wireframe(
            grid_r[:, :, 0],
            grid_R[:, :, 0],
            dVdr[:, :, 0],
            rstride=stride,
            cstride=stride
            )
    ax.plot_wireframe(
            grid_r[:, :, 0],
            grid_R[:, :, 0],
            dVdrAnalytic[:, :, 0],
            rstride=strideAna,
            cstride=strideAna,
            color='r'
            )
    dVdR = derivative(lambda x: potential(grid_r, x, grid_cosGamma), grid_R)
    dVdRAnalytic = -alpha*potential(grid_r, grid_R, grid_cosGamma)
    ax = fig.add_subplot(132, projection='3d')
    ax.plot_wireframe(
            grid_r[:, :, 0],
            grid_R[:, :, 0],
            dVdR[:, :, 0],
            rstride=stride,
            cstride=stride
            )
    ax.plot_wireframe(
            grid_r[:, :, 0],
            grid_R[:, :, 0],
            dVdRAnalytic[:, :, 0],
            rstride=strideAna,
            cstride=strideAna,
            color='r'
            )
    dVdcosGamma = derivative(
            lambda x: potential(grid_r, grid_R, x),
            grid_cosGamma
            )
    dVdcosGammaAnalytic = 3.*beta*C*np.exp(-alpha*grid_R)*grid_cosGamma
    ax = fig.add_subplot(133, projection='3d')
    ax.plot_wireframe(
            grid_cosGamma[:, 0, :],
            grid_R[:, 0, :],
            dVdcosGamma[:, 0, :],
            rstride=stride,
            cstride=stride
            )
    ax.plot_wireframe(
            grid_cosGamma[:, 0, :],
            grid_R[:, 0, :],
            dVdcosGammaAnalytic[:, 0, :],
            rstride=strideAna,
            cstride=strideAna,
            color='r'
            )
    plt.legend(['numeric', 'analytic'])
    plt.show()
    return


if __name__ == "__main__":
    main(sys.argv[1:])
