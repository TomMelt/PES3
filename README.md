Scatter PES
===========

3-body classical scattering code on a quantum PES.

![Schematic](img/Schematic.png)

Please clone the repo:

    git clone git@github.com:TomMelt/ScatterPES.git

Installation
============

To install on local machine:

    pip install -U -e .

Core Dependencies
=============
These package must be installed for ```scatter``` to run properly.

* ```numpy```
* ```scipy```
* ```xarray```

Plotting Dependencies
=============
Only required for the plotting functions.

* ```matplotlib```

To Do
=====
* Take a look at the classical action (zeroth order phase)
    * if it varies slowly it indicates classical nature
    * if it varies rapidly calculate the phase and interferance for quantum number n
    * classical action of whole system or of pbar?
* still need to write tests for:
    * ```classify.py```
    * ```initialize.py```
    * ```propagation.py```
* Check how starting position affects the scattering velocities/ distro of nq
    * Could calculate the integral of energy gained by moving through 1/r pot.

Notes
=====
* initially program was also built in spherical coords
    - this does not work well with RK4 prop because angles can be arbitrarily large and;
    - there are singularities in the equations of motion (1/sin^2(phi)) etc.
* Error in the Hamiltonian appears to come from close encounter of all three particles
    - tried using just short range potential < 5 a.u. but this was not successful
* E.E.Nikitin Jost W. (Ed.), Phys. chem., 6a, Academic Press, New York (1974) ch. 4
    - "one of the necessary conditions for the proper description in terms of a classical traj. is that changes in the De Broglie wavelength of the appropriate vars. be small over the scale determined by the spatial variation of the wavefunction"
    - "epsilon*b*b = l_c^2/2Mu" 
