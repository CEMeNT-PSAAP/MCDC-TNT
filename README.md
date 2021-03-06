# MC/DC-TNT
**Monte Carlo / Dynamic Code - Toy Neutronics Testbed**

We seek to explore acceleration schemes from Python to see if we can implement performance portable and easier to produce code for a Monte Carlo neutron transport code: [MC/DC](https://github.com/CEMeNT-PSAAP/MCDC). MC/DC is the ultimate deliverable this code is just for testing methods of implementation

## Current Kernel Status
0. Pure Python: working transient
1. Numba CPU: working transient
2. PyOPM CPU: working transient
3. Numba CUDA: working transient
4. Pykokkos: working transient
5. PyCUDA: working static* 
6. PyOpenCL: working static*

*static kernels might fail

## Methods of Implementation
We explore three methods to implement our transport functions on:
1. [PyKokkos](https://github.com/kokkos/pykokkos) portability framework;
2. Hardware code generating libraries [PyOpenCL](https://documen.tician.de/pyopencl/) and [PyCUDA](https://wiki.tiker.net/PyCuda/Installation/Linux/) which have been implemented before in  [PyFR](https://github.com/PyFR/PyFR) (tho their use has been depricated there); and
3. [Numba](https://numba.pydata.org/)

## Implemented Physics
The neutronic physics we target are:
1. mono-energetic;
2. multi-region (slab geometry);
3. particles produced from fissions;
4. surface tracking and track length estimators; and
5. transient modeling

## Quick Setup
I don't expect anyone to run this but for those who are interested; have at it. Find all my embarsing mistakes! This process will get you the dependancies required for Numba (CPU/GPU) and pure Python. *NOTE:* For Numba CUDA you must have **(a)** a [proper graphics card](https://numba.pydata.org/numba-doc/latest/cuda/overview.html#) and; **(b)** the correct [drivers installed](https://numba.pydata.org/numba-doc/latest/cuda/overview.html#) and in your path. Other than that the process is pretty stright forward:
1. From a terminal with conda installed set up a conda environment with `conda create -n mcdc_tnt numba matplotlib pyyaml pytest` which will install all package dependencies for the Numba (no pyomp) and Pure Python implementations
2. run `conda activate mcdc_tnt`
3. Clone this github 
4. Run `pip install --user -e .` in project directory to install mcdc_tnt as a local package.
5. Run `python run.py -i tc_1.yaml` for a numba threading 
6. To change to GPU or pure Python appened the run command with `-t nb_gpu` or `-t pp` respectively

## More complicated Installations
Warning: *Here be dragons!* Unfortunately some of these packages are not only complicated to build but also conflicting meaning the conda environment manager is required to be able to switch back and forth. Note that these builds all have there own required dependencies and potentialy there own bugs that might be exposed on your particular system with your particular drivers.

### Installation of PyKokkos
This is subject to change and should be compared to the PyKokkos build instructions but this is what it looked like for me. Cation is reqired when using various versions of CUDA (10.2), gcc (8 or 9), and cmake (16). If you don't have acess to a machine that has a `module` system then check out [update-alternatives](https://linuxconfig.org/how-to-switch-between-multiple-gcc-and-g-compiler-versions-on-ubuntu-20-04-lts-focal-fossa) to switch things that cannot be installed via conda (gcc). While most machines should be able to operate with the OpenMP backend, I have gotten the Lassen Machine to install the CUDA version. If you don't want to touch the more difficult CUDA instiliation then switch to the OpenMP only version change  `-DENABLE_CUDA` from `ON` to `OFF`.

1. `git clone` [`pykokkos-base`](https://github.com/kokkos/pykokkos-base) and the develop branch of [`pykokkos`](github.com/kokkos/pykokkos). To do this in the pykokkos directory run `git fetch` then `git checkout develop`
2. Prep conda environment by snagging requirements listed in requirements.txt from pykokkos-base and pykokkos. (1. `conda create -n pyk` 2. `conda activate pyk` 3. in pyk-base directory run `conda install --file requirments.txt` 4. in pykokkos directory run `conda install --file requirments.txt`) *ensure that cmake is of version 18 or higher and that gcc/g++ versions are at least 9*
3. Install Pykokkos-base for both OpenMP, and CUDA implementations by running (OpenMP much quicker and seems to have less issues, if that's all you need switch `-DENABLE_CUDA=OFF`):
`python setup.py install -- -DCMAKE_CXX_COMPILER=g++ -DENABLE_LAYOUTS=ON -DENABLE_MEMORY_TRAITS=OFF -DENABLE_VIEW_RANKS=3 -DENABLE_CUDA=ON -DENABLE_THREADS=OFF -DENABLE_OPENMP=ON -G "Unix Makefiles" -- -j 4` *this will take upwards of 2 hours to build and will consume a considerable ammount of RAM*
4. Install pykokkos using `pip install --user -e .`
5. Run!

### Installation of PyOpenMP
This process was described in a [paper](https://ieeexplore.ieee.org/document/9658236) that the larger prgoraming framework was described in
1. Make a new conda environment seperate from the normal one in which you have installed the current release of Numba
2. Install all dependinces and packages with `conda install numba -c drtodd13 -c conda-forge --override-channels`
3. Install other MCDC-TNT requirements (pyyaml, matplotlib)

### Installation of PyCUDA
1. Follow [PyCUDA documentation](https://wiki.tiker.net/PyCuda/Installation/Linux/)
2. Install other MCDC-TNT requirements (pyyaml, matplotlib, numba) (only the advance function is implemented via this method for hcgls)

### Installation of PyOpenCL
1. Follow [PyOpenCL documentation](https://documen.tician.de/pyopencl/misc.html)
2. Install other MCDC-TNT requirements (pyyaml, matplotlib, numba)


## Interface
This project is designed to be interfaced with via the command line and an input.yaml file. An example is listed here:

```
name: 'fissioning_slab'   #name of the simluation (any string)
number of particles: 1e5  #number of particles top initiate in the 
rng seed: 777             #random number seed (int)
particle speed: 1         #particle speed (float)
neutrons per fission: 2   #how many neutrons to produce per fission event
isotropic: Ture           #isotropic source? if true than particles produced with a random direction

length of slab: 1         #width of the slab
surface locations: [0,1]  #region geometry deffitinition (vector of floats)


mesh mod generation: n #n for number of desieredd cells, dx for desiered cell width)
dx: 0.01   #mesh width (for error and scalar flux tracking) (float)
n: 81


hardware target: nb_cpu          #specifying the hardware target: pp/nb_cpu/nb_gpu/pyk_cpu/pyk_gpu
print warmup times: True         #print warm up times

assemble mesh: True             #assemble mesh from crossections listed here
capture cross section: 0.333    #should be as many values here as regions specified in surface_locations
scatter cross section: 0.333
fission cross section: 0.333

transient tally: True           #to see transient behavior must be true
tally dt: 1                     #time inbetween census snap shots
max time: 20                    #time of simulation end

file output: True               #should it output flux and stats? if a special file name is desiered supple in command line

error plot: True                #produce the error plot?
flux plot: True                 #produce the flux plot?

#Issues might arries for gpus without the ability to do 64bit math
#note that native python "float" is a 64bit
floating point accuracy: float #float (np.float32) double (np.float64)

phase space vec: 4              #defining actual static size of phase space vectors

```

Then to run a simulation it can be done from a python file using:
```
import mcdc_tnt
mcdc_tnt.run('input.yaml','output.out','hardware_target')
```

or from the command line in the mcdc_tnt directory with:
`python run.py -i input.yaml -o output.out -t 'hardware_target'`


*For thoes about to rock, we salute you*
