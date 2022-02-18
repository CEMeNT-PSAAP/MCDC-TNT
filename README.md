# MC/DC-TNT
**Monte Carlo / Dynamic Code - Toy Neutronics Testbed**: Is there an easier way for HPC development? Free from the seg-faults and semi-colons? ¯\\_(ツ)_/¯ Lets find out!

We seek to explore software engineering methods from Python to see if we can implement performance portable and *easier* to produce code for a Monte Carlo neutron transport code.

## Methods of Implementation
We explore three mothdes to implement our transport functions on:
1. [PyKokkos](https://github.com/kokkos/pykokkos) portability framework;
2. [Mako templates with hardware code generating libraries](https://github.com/PyFR/PyFR); and
3. [Numba](https://numba.pydata.org/)

## Implemented Physics
The neutronic physics we target are:
1. monoenergetic;
2. multiregion (slab geometry);
3. particles produced from fissions;
4. surface tracking and track length estimators; and
5. trainsent modeling
For more information about the impolemented neutron transport throy see the rundown in the docs folder.

*For thoes about to rock, we salute you*
