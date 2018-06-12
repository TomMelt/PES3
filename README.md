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
* check error in H relative to scattering energy (comes from IF switch)
    - run PES_fix.f to check different sols
    - error appears to be on close encounter of all three particles
    - tried using just short range potential < 5 a.u. but this was not successful
* check PEC goes to zero at infinity (used in classify.py)
    - changed diatom classify to correct asymptotic value of diatomPEC
* fix nquantum to the correct reduced mass
