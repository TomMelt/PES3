from pk import PK as potential
from scatter.transform import internuclear
import scatter.constants as c


def Hamiltonian(r, p, R, P):
    R1, R2, R3 = internuclear(r, R)

    H = p@p/(2.*c.mu) + P@P/(2.*c.MU) + potential(R1, R2, R3)
    return H
