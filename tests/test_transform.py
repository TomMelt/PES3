import pytest
import numpy as np
from scatter import (getPropCoords, getParticleCoords, sphericalToCart,
                        internuclear)


def test_getPropCoords():
    r, p, R, P = getPropCoords(
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            )
    assert all(r == np.array([1, 2, 3]))
    assert all(p == np.array([4, 5, 6]))
    assert all(R == np.array([7, 8, 9]))
    assert all(P == np.array([10, 11, 12]))

    with pytest.raises(IndexError):
        r, p, R, P = getPropCoords(
                np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                )

    with pytest.raises(IndexError):
        r, p, R, P = getPropCoords(
                np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
                )


def test_getParticleCoords():
    r = np.array([1, 2, 3])
    R = np.array([3, 0, -4])
    p = -np.array([1, 2, 3])
    P = np.array([-3, 4, 1])
    rfail = np.array([1, 2])
    # TODO: write a test to check the actual maths
    r1, r2, r3, p1, p2, p3 = getParticleCoords(r, p, R, P)

    with pytest.raises(IndexError):
        r1, r2, r3, p1, p2, p3 = getParticleCoords(rfail, p, R, P)


def test_sphericalToCart():
    r = 5.
    theta = np.pi
    phi = 30./180.*np.pi
    ans = sphericalToCart(r, theta, phi)
    actual = np.array([-5./2., 0., 5.*np.sqrt(3.)/2.])
    for i, x in enumerate(ans):
        assert x == pytest.approx(actual[i])
    with pytest.raises(ValueError):
        sphericalToCart(-1., theta, phi)
    with pytest.raises(ValueError):
        sphericalToCart(r, 2.3*np.pi, phi)
    with pytest.raises(ValueError):
        sphericalToCart(r, -0.1, phi)
    with pytest.raises(ValueError):
        sphericalToCart(r, theta, 1.3*np.pi)
    with pytest.raises(ValueError):
        sphericalToCart(r, theta, -0.1)
    with pytest.raises(TypeError):
        sphericalToCart(r, theta, int(phi))


def test_internuclear():
    r = np.array([3, 4, 5])
    R = np.array([5, -2, 9])
    R1, R2, R3 = internuclear(r, R)
    assert R1 == pytest.approx(5.*np.sqrt(2.))
    assert R2 == pytest.approx(np.sqrt(349./2.))
    assert R3 == pytest.approx(np.sqrt(141./2.))
    with pytest.raises(TypeError):
        R1, R2, R3 = internuclear(r, [1., -3., 4.])
    with pytest.raises(IndexError):
        internuclear(np.array([1, 2, 3, 4]), R)
    with pytest.raises(IndexError):
        internuclear(r, np.array([1, 2, 3, 4]))
