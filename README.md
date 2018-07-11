PES Scatter
===========

3-body classical scattering code on a quantum PES.

Please clone the repo:

    git clone git@github.com:TomMelt/PES3.git

To install on local machine:

    pip install -U -e .

Prerequisites
* ```numpy```
* ```random```
* ```scipy```

TODO:
====
* write tests.
* change transform.getParticleCoords() to be in lab frame coordinates

NOTE:
=====
* initially program was also built in spherical coords
    - this does not work well with RK4 prop because angles can be arbitrarily large and;
    - there are singularities in the equations of motion (1/sin^2(phi)) etc.
* Error in the Hamiltonian appears to come from close encounter of all three particles
    - tried using just short range potential < 5 a.u. but this was not successful
* E.E.Nikitin Jost W. (Ed.), Phys. chem., 6a, Academic Press, New York (1974) ch. 4
    - "one of the necessary conditions for the proper description in terms of a classical traj. is that changes in the De Broglie wavelength of the appropriate vars. be small over the scale determined by the spatial variation of the wavefunction"
    - "epsilon*b*b = l_c^2/2Mu" 
