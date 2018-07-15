import pytest
import numpy as np
from scatter import diatomPEC, potential, Hamiltonian
from scatter.numeric import (pesWrapper, derivative, numericDerivatives,
        equation_of_motion)
from scatter import constants as c


def test_pesWrapper():
    R1 = 5.
    R2 = 10.
    R3 = 20.

    ans = pesWrapper(R1, R2, R3)
    assert isinstance(ans, float)

    with pytest.raises(TypeError):
        pesWrapper(1, R2, R3)

    with pytest.raises(ValueError):
        pesWrapper(R1, R2, -R3)


def test_diatomPEC():
    R1 = 1.

    ans = diatomPEC(R1)
    assert isinstance(ans, float)

    with pytest.raises(TypeError):
        diatomPEC(1)

    with pytest.raises(ValueError):
        diatomPEC(-R1)


def test_potential():
    R1 = 1.
    R2 = 109.
    R3 = 8.

    ans = potential(R1, R2, R3)
    assert isinstance(ans, float)

    with pytest.raises(TypeError):
        potential(1, R2, R3)

    with pytest.raises(ValueError):
        potential(R1, -R2, R3)


def test_Hamiltonian():
    r = np.array([1, 2, 3])
    R = np.array([3, 0, -4])
    p = -np.array([1, 2, 3])
    P = np.array([-3, 4, 1])
    rfail = np.array([1, 2])

    assert isinstance(Hamiltonian(
        r=r, p=p, R=R, P=P), float
        )

    with pytest.raises(IndexError):
        Hamiltonian(r=rfail, p=p, R=R, P=P)


def test_numeric():
    for method in ["stencil", "euler"]:
        func = lambda x: x*x
        ans = derivative(func=func, x=3., method=method)
        assert ans == pytest.approx(6.)

        func = lambda x: np.sin(x)
        ans = derivative(func=func, x=3., method=method, h=1e-06)
        assert ans == pytest.approx(np.cos(3.))


def test_numericDerivatives():
    r = np.array([1, 2, 3])
    R = np.array([3, 0, -4])
    rfail = np.array([1, 2])

    pdot, Pdot = numericDerivatives(r=r, R=R)

    for ans in [pdot, Pdot]:
        assert isinstance(ans, np.ndarray)
        assert len(ans) == 3

    with pytest.raises(IndexError):
        numericDerivatives(r=rfail, R=R)


def test_equation_of_motion():
    t = 10.
    coords = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    with pytest.raises(TypeError):
        equation_of_motion(
                t=int(t),
                coordinates=coords
                )

    with pytest.raises(IndexError):
        equation_of_motion(
                t=t,
                coordinates = coords[:-1]
                )

    ans = equation_of_motion(t, coords)

    assert len(ans) == c.dim
    assert isinstance(ans, np.ndarray)
