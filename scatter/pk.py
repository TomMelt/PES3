from numpy import square as pow2
from numpy import sqrt, exp

D1 = 0.174436808
D3 = 0.072279592
d = 1.036345596
e = -0.643122266
eH2 = -0.174437

Re = 1.40083
a = 1.04435
b = 1.000122
k = 0.60
lamb = 0.65


def PK(R1, R2, R3):

    if R1 < 0.2 or R2 < 0.2 or R3 < 0.2:
        PK = 1.1
        return PK
    else:

        S1 = S(R1)
        S2 = S(R2)
        S3 = S(R3)

        J1 = J(R1, R3, R2)
        J2 = J(R2, R1, R3)
        J3 = J(R3, R1, R2)

#        print(J1)
#        print(J2)
#        print(J3)

        Q = Qd(R1) + Qd(R2) + Qd(R3)

        print(Q)

        S123 = S1*S2*S3
        J123 = e*S123

        S11 = S1*S1
        S22 = S2*S2
        S33 = S3*S3

        C1 = pow2(1.-S123)-0.5*(pow2(S11-S22)+pow2(S22-S33)+pow2(S11-S33))
        C2 = -(Q-J123)*(1.-S123)+0.5*((J1-J2)*(S11-S22)+(J2-J3)*(S22-S33)+(J1-J3)*(S11-S33))
        C3 = pow2(Q-J123)-0.5*(pow2(J1-J2)+pow2(J2-J3)+pow2(J1-J3))

        PK = 1./C1*(-C2-sqrt(C2*C2-C1*C3)) - eH2
        return PK


def S(R):
    S = (1. + F(R)*R + 1./3.*pow2(F(R)*R))*exp(-F(R)*R)
    return S


def F(R):
    F = 1. + k*exp(-lamb*R)
    return F


def Qd(R):
    Qd = 0.5*(E1(R) + E3(R) + pow2(S(R))*(E1(R) - E3(R)))
    return Qd


def E1(R):
    E1 = D1*(exp(-2.*a*(R-Re)) - 2.*exp(-a*(R-Re)))
    return E1


def E3(R):
    E3 = D3*(exp(-2.*b*(R-Re)) + 2.*exp(-b*(R-Re)))
    return E3


def J(Ra, Rb, Rc):
    J1 = 0.5*(E1(Ra) - E3(Ra))
    J2 = 0.5*(E1(Ra) + E3(Ra))
    J3 = (1. + (1./Rb))*exp(-2.*Rb)
    J4 = (1. + (1./Rc))*exp(-2.*Rc)
    J = J1 + pow2(S(Ra))*(J2 + d*(J3 + J4))
    return J


if __name__ == "__main__":
    print(PK(1., 2., 3.))
