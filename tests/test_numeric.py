import pytest
import numpy as np
import scatter
from scatter import (pesWrapper, diatomPEC, potential, Hamiltonian,
        derivative)


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
