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
* check PEC goes to zero at infinity (used in classify.py)
* fix nquantum to the correct reduced mass
