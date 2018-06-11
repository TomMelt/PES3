from .initialize import rovibrationalEnergy
from mpl_toolkits.mplot3d import Axes3D # noqa
from .numeric import potential, diatomPEC
from .propagation import assignClassical
from .transform import getPropCoords, getParticleCoords, internuclear
from . import constants as c
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as itp


def func(x, y, R1):
    R2 = np.sqrt(x*x+y*y+1.-2*y)
    R3 = np.sqrt(x*x+y*y+1.+2*y)
    return potential(R1, R2, R3)


def testPotential():
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.linspace(-3, 3, num=50)
    Y = np.linspace(-3, 3, num=50)
    X, Y = np.meshgrid(X, Y)
    Z = func(X, Y, 2.)

    # Plot the surface.
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)

    ax.set_zlim([-3., 1.])

    plt.show()

    X = np.linspace(0.5, 15., num=100)

    plt.plot(X, diatomPEC(X) - diatomPEC(2.), '-')
    for v in range(5):
        plt.plot(
                [1., 4.],
                [rovibrationalEnergy(v, 0), rovibrationalEnergy(v, 0)],
                '-k'
                )
    ax.set_ylim([0, 2])
    plt.show()

    R1 = np.linspace(1.0, 20., num=100)
    R2 = R1

    plt.plot(X, potential(2., R1, R2), '-')

    plt.show()


def plotKE(data):

    fig = plt.figure()

    KE1, KE2 = [], []
    H = []
    R1, R2, R3 = [], [], []

    for row in data:
        r, p, R, P = getPropCoords(row[1:-1])
        KE1.append(p@p/(2.*c.mu))
        KE2.append(P@P/(2.*c.MU))
        H.append(row[-1])
        r1, r2, r3 = internuclear(r, R)
        R1.append(r1)
        R2.append(r2)
        R3.append(r3)

    KE1 = np.array(KE1)
    KE2 = np.array(KE2)
    H = np.array(H)

    ax = fig.add_subplot(221)
    ax.set_xlabel('time (a.u.)')
    ax.set_ylabel(r'$E$ (a.u.)')
    ax.plot(data[:, 0], KE1, '-k')
    ax.plot(data[:, 0], KE2, '-r')
    ax.plot(data[:, 0], H, '-g')
    ax.legend([r'$E_k^1$', r'$E_k^2$', r'$H_{system}$'])

    ax = fig.add_subplot(222)
    ax.set_xlabel('time (a.u.)')
    ax.set_ylabel(r'$E$ (a.u.)')
    ax.plot(data[:, 0], H, '.g')
    ax.legend([r'$H_{system}$'])

    ax = fig.add_subplot(223)
    ax.set_xlabel('time (a.u.)')
    ax.set_ylabel(r'$R$ (a.u.)')
    ax.plot(data[:, 0], R1, '.k')
    ax.plot(data[:, 0], R2, '.b')
    ax.plot(data[:, 0], R3, '.r')
    ax.legend([r'$R_{BC}$', r'$R_{AB}$', r'$R_{AC}$'])
    plt.show()

    ax = fig.add_subplot(224)
    ax.set_xlabel('time (a.u.)')
    ax.set_ylabel(r'$V$ (a.u.)')
    ax.plot(data[:, 0], R3, '-r')
    ax.plot(data[:, 0], potential(R1, R2, R3), '-k')
    ax.legend([r'$V{(R1, R2, R3}$'])

    return


def plot3Danim(data, tstart=0., tfinal=None, tstep=10.):

    x1, x2, x3 = [], [], []
    for row in data:
        r, p, R, P = getPropCoords(row[1:-1])
        r1, r2, r3, p1, p2, p3 = getParticleCoords(r, p, R, P)
        x1.append(r1)
        x2.append(r2)
        x3.append(r3)
    x1 = np.array(x1)
    x2 = np.array(x2)
    x3 = np.array(x3)
    ta = data[:, 0]
    if tfinal is None:
        tfinal = ta[-1]
    tn = np.arange(tstart, tfinal, step=tstep)
    interpx1 = itp.interp1d(ta, x1.T)
    interpx2 = itp.interp1d(ta, x2.T)
    interpx3 = itp.interp1d(ta, x3.T)
    x1 = interpx1(tn).T
    x2 = interpx2(tn).T
    x3 = interpx3(tn).T

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_zlim(-5, 45)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    for i in range(len(tn)):
        ax.scatter(x1[i, 0], x1[i, 1], x1[i, 2], c='r', marker='.')
        ax.scatter(x2[i, 0], x2[i, 1], x2[i, 2], c='b', marker='.')
        ax.scatter(x3[i, 0], x3[i, 1], x3[i, 2], c='k', marker='.')
        plt.pause(0.001)
    plt.ioff()
    plt.show()

    return


def plot3Dtrace(data):

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_zlim(-5, 45)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    x1, x2, x3 = [], [], []
    for row in data[0::1]:
        r, p, R, P = getPropCoords(row[1:-1])
        r1, r2, r3, p1, p2, p3 = getParticleCoords(r, p, R, P)
        x1.append(r1)
        x2.append(r2)
        x3.append(r3)
    x1 = np.array(x1)
    x2 = np.array(x2)
    x3 = np.array(x3)
    ax.plot(x1[:, 0], x1[:, 1], x1[:, 2], '-r')
    ax.plot(x2[:, 0], x2[:, 1], x2[:, 2], '-b')
    ax.plot(x3[:, 0], x3[:, 1], x3[:, 2], '-k')
    plt.show()

    return


def plotClassical(data):

    fig = plt.figure()

    R1, R2, R3 = [], [], []
    Ec, lc = [], []

    t = data[:, 0]

    for row in data:
        r, p, R, P = getPropCoords(row[1:-1])
        r1, r2, r3 = internuclear(r, R)
        R1.append(r1)
        R2.append(r2)
        R3.append(r3)

        assignment = assignClassical(r=r, R=R, p=p, P=P)
        lc.append(assignment["lc"])
        Ec.append(assignment["Ec"])

    ax = fig.add_subplot(121)
    ax.set_xlabel('time (a.u.)')
    ax.set_ylabel(r'$E_c$ (a.u.)')
    ax.plot(t, Ec, '-k')
    ax.legend([r'$E_{c}$'])
    plt.show()

    ax = fig.add_subplot(122)
    ax.set_xlabel('time (a.u.)')
    ax.set_ylabel(r'$R$ (a.u.)')
    ax.plot(t, lc, '-k')
    ax.legend([r'$L_{c}$'])
    plt.show()

    return
