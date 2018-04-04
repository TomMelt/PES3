import pytest # noqa
import numpy as np
from scatter import getParticleCoords
from scatter.main import initialiseScattering, initialiseRotor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # noqa


def testInitialScatter(rand, epsilon):

    print("test: calculating initial scattering particle conditions")

    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    data = []

    for i in range(1000):
        R, P = initialiseScattering(rand, epsilon)
        data.append(R[0])
        r1, r2, r3 = getParticleCoords(R, P, R, P)
        ax.scatter(r3[0], r3[1], r3[2], c='k', marker='.')

    data = np.array(data)
    ax = fig.add_subplot(122)
    plt.hist(data, 50)

    plt.show()

    assert 1 == 1


def testInitialRotor(rand):

    print("test: calculating initial rotor conditions")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    for i in range(1000):
        r, p = initialiseRotor(rand)
        r1, r2, r3 = getParticleCoords(r, p, r, p)
        ax.scatter(r1[0], r1[1], r1[2], c='r', marker='.')
        ax.scatter(r2[0], r2[1], r2[2], c='b', marker='.')

    plt.show()

    assert 1 == 1
