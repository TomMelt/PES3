h2+pbar potential energy surface
--------------------------------

Source code description:

* ```emax.h``` is paramaters for neural network fit
* ```NNPARM.DATA125``` is neural network parameters
* ```H2HBAR-pot-nn.f``` is neural network function
* ```PES3.f``` is combined pec fit

Use ```f2py``` to convert fortran code to python module using:

    f2py -c --fcompiler=gnu95 -m pes PES3.f H2+pbar-bpm-pot-nn.f

After compilation/conversion move the ```.so``` module

    mv pes.cpython-36m-x86_64-linux-gnu.so ../pes.so
