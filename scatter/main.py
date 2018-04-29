from scatter.plot import plotKE, plot3Dtrace # noqa
from scatter.transform import getPropCoords, internuclear, distanceFromCoM, getParticleCoords
import numpy as np
import random as rnd
import scatter.analytics as ana
import scatter.constants as c
import scatter.numeric as num
import scipy.integrate as odeint
import sys
# import xarray as xr


# TODO:
# - add quantum classification after end of each trajectory
# - add/fix tests
# - check morse potential consts for H2 diatom
# - should R+/- (initial diatom) be selected uniformly?
# - change definition of reaction (not inelastic/elastic)
#   instead it is whether AC+B or AB+C form from BC+A

# NOTE:
# ----------------------------------------------------------------------------
# - I have checked the rovibrational energy levels in the numeric.py
#   function rovibrationalEnergy(v, J). They agree with literature.
#   see the text file comparison_rovibrational_h2.txt
# ----------------------------------------------------------------------------
# - I have checked the initial conditions for the diatom
#   inner(ri,pi) == 0 by definition
# ----------------------------------------------------------------------------
# - initially program was also built in spherical coords but this does not
#   work well with RK4 prop because angles can be arbitrarily large
#   and there are singularities in the equations of motion (1/sin^2(phi)) etc.
# ----------------------------------------------------------------------------
# - "one of the necessary conditions for the proper description in terms of a
#   classical traj. is that changes in the De Broglie wavelength of the
#   appropriate vars. be small over the scale determined by the spatial
#   variation of the wavefunction"
#   And "epsilon*b*b = l_c^2/2Mu"
#   E.E.Nikitin
#   Jost W. (Ed.), Phys. chem., 6a, Academic Press, New York (1974)
#   ch. 4
# ----------------------------------------------------------------------------


def isConverged(dist):
    c1 = dist > c.R0
    return c1


def main(args):

    seed = int(args[0])
    rand = rnd
    rand.seed(seed)

    epsilon = float(args[1])
    v = float(args[2])
    J = float(args[3])

    data = []

    countElastic = 0
    countTotal = 0
    countAB, countAC, countBC = 0, 0, 0

    for i in range(400):

        # initial conditions of the scattering particles
        ri, pi = num.initialiseDiatomic(v, J, rand)
        Ri, Pi = num.initialiseScattering(rand, epsilon)
        b = Ri[0]
        initialConditions = np.concatenate((ri, pi, Ri, Pi), axis=0)

        # initialise stepping object
        stepper = odeint.RK45(
                lambda t, y: num.equation_of_motion(t, y),
                c.ts, initialConditions, c.tf,
                max_step=c.maxstep, rtol=c.rtol, atol=c.atol
                )

        # array to store trajectories
        r, p, R, P = getPropCoords(stepper.y)
        KE1i = p@p/(2.*c.mu)
        KE2i = P@P/(2.*c.MU)
        H = ana.Hamiltonian(r, p, R, P)
        Hi = H
        trajectory = []
        trajectory.append([stepper.t] + stepper.y.tolist() + [H])
        dist = 0.
        maxstep, maxErr = 0., 0.
        countstep = 0

#        if i != 47: continue

        # propragate Hamilton's eqn's
        while stepper.t < c.tf:
            try:

                stepper.step()

                r, p, R, P = getPropCoords(stepper.y)
                R1, R2, R3 = internuclear(r, R)
                r1, r2, r3, p1, p2, p3 = getParticleCoords(r, p, R, P)

                if R1 == np.max([R1, R2, R3]):
                    dist = distanceFromCoM(c.m2, c.m3, r2, r3, r1)
                if R2 == np.max([R1, R2, R3]):
                    dist = distanceFromCoM(c.m1, c.m3, r1, r3, r2)
                if R3 == np.max([R1, R2, R3]):
                    dist = distanceFromCoM(c.m1, c.m2, r1, r2, r3)

                trajectory.append([stepper.t] + stepper.y.tolist() + [H])

                maxstep = np.max([stepper.step_size, maxstep])
                maxErr = np.max([np.abs(H - ana.Hamiltonian(r, p, R, P))/H, maxErr])
                countstep = countstep + 1

                if isConverged(dist):
                    KE1f = p@p/(2.*c.mu)
                    KE2f = P@P/(2.*c.MU)
                    Hf = H
                    countTotal += 1
                    if R1 == np.min([R1, R2, R3]):
                        countBC += 1
                    if R2 == np.min([R1, R2, R3]):
                        countAB += 1
                    if R3 == np.min([R1, R2, R3]):
                        countAC += 1
                    if np.abs(KE2i - KE2f) < c.cutoff:
                        countElastic += 1
                    break
                H = ana.Hamiltonian(r, p, R, P)
            except RuntimeError as e:
                print(e)
                break
        trajectory = np.array(trajectory)

#        plotKE(trajectory)
#        plot3Dtrace(trajectory)

        if i % 10 == 0:
            print(i)

        data.append(
                [seed, int(J), int(v), i, epsilon]
                + [b, c.R0, dist]
                + list(ri) + list(pi)
                + list(Ri) + list(Pi)
                + list(r) + list(p)
                + list(R) + list(P)
                + [R1, R2, R3]
                + [KE1i, KE2i, KE1f, KE2f]
                + [stepper.t, countstep, maxstep]
                + [maxErr, Hi, Hf]
                + [countElastic, countTotal]
                + [countAB, countAC, countBC])

    print(epsilon, np.pi*c.bmax*c.bmax*float(countAB+countAC)/float(countTotal))

    outfile = open("data"+str(int(J))+str(int(v))+str(seed)+".csv", "w")
    for line in data:
        outfile.write(' '.join(str(e) for e in line)+"\n")

    return


if __name__ == "__main__":
    main(sys.argv[1:])
