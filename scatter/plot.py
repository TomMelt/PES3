from mpl_toolkits.mplot3d import Axes3D # noqa
from scatter.transform import getPropCoords, getParticleCoords
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import scatter.constants as c
import pk
from numeric import rovibrationalEnergy, rootFunctional


def testRovibrationalEnergy():
    plt.figure()
    for J in range(10):
        for v in range(5):
            plt.plot([J], [rovibrationalEnergy(v, J)], 'kx')
    plt.show()
    return


def testRootFunctional():
    plt.figure()
    r = np.linspace(1., 2., 100)
    y = rootFunctional(0, 0, r)
    plt.plot(r, y, '-k')
    plt.show()
    f = lambda x: rootFunctional(0, 0, x)
    print(f(1.), f(pk.Re))
    print(opt.brentq(lambda x: rootFunctional(0, 0, x), 0.1, pk.Re))
    print(opt.brentq(lambda x: rootFunctional(0, 0, x), 3., pk.Re))
    return


def plotKE(data):

    fig = plt.figure()

    KE1, KE2 = [], []
    H = []

    for row in data:
        r, p, R, P = getPropCoords(row[1:-1])
        KE1.append(p@p/(2.*c.mu))
        KE2.append(P@P/(2.*c.MU))
        H.append(row[-1])

    KE1 = np.array(KE1)
    KE2 = np.array(KE2)
    H = np.array(H)

    ax = fig.add_subplot(221)
    ax.set_xlabel('time (a.u.)')
    ax.set_ylabel(r'E_k')
    ax.plot(data[:, 0], KE1, '-k')
    ax.plot(data[:, 0], KE2, '-r')
    ax.plot(data[:, 0], H, '-g')
    plt.show()

    return


def plotStats(data):
    data = np.array(data)
    fig = plt.figure(figsize=(3, 3))
    fig.add_subplot(221, title='b vs KE1')
    plt.hist2d(data[:, 0], data[:, 1], bins=20)
    fig.add_subplot(222, title=f'1D b (bmax = {c.bmax})')
    plt.hist(data[:, 0], 10)
    fig.add_subplot(223, title='1D KE1')
    plt.hist(data[:, 1], 10)
    fig.add_subplot(224, title='cross-section')
    plt.plot(data[:, 2])

    plt.show()
    return


def plot3Dtrace(data):

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    for row in data[0::10]:
        r, p, R, P = getPropCoords(row[1:-1])
        r1, r2, r3, p1, p2, p3 = getParticleCoords(r, p, R, P)
        ax.scatter(r1[0], r1[1], r1[2], c='r', marker='.')
        ax.scatter(r2[0], r2[1], r2[2], c='b', marker='.')
        ax.scatter(r3[0], r3[1], r3[2], c='k', marker='.')
        plt.pause(0.1)
    plt.ioff()
    plt.show()
    del fig

    return
