import numpy as np
import matplotlib.pyplot as plt

sf_actual = np.loadtxt('anwser.pout', comments='#', delimiter=',', skiprows=2)
sf_pp = np.loadtxt('pp.out', comments='#', delimiter=',', skiprows=2)
sf_nbc = np.loadtxt('numba_cpu.out', comments='#', delimiter=',', skiprows=2)
sf_pyomp = np.loadtxt('numba_cpu_pyomp.out', comments='#', delimiter=',', skiprows=2) 
sf_pykc = np.loadtxt('pyk_cpu.out', comments='#', delimiter=',', skiprows=2) 

def error(sim, bench):
    error = np.linalg.norm(sim - bench) / np.linalg.norm(bench)
    return(error)
    
print('Produced Errors Between Soultions')
print('     -pure python............{0}'.format(error(sf_actual, sf_pp)))
print('     -numba threading........{0}'.format(error(sf_actual, sf_nbc)))
print('     -numba pyomp............{0}'.format(error(sf_actual, sf_pyomp)))
print('     -pyk ompenmp............{0}'.format(error(sf_actual, sf_pykc)))
print()

plt.figure(1)
f = plt.plot(sf_actual[:,1], sf_actual[:,2], '-b',
         sf_pp[:,1], sf_pp[:,2], '-r',
         sf_nbc[:,1], sf_nbc[:,2], 'g--',
         sf_pyomp[:,1], sf_pyomp[:,2], 'y--',
         sf_pykc[:,1], sf_pykc[:,2], 'k--')
plt.title("Scalar Flux")
plt.ylabel("$\phi [cm^{-2}s^{-1}]$")
plt.xlabel("x [cm]")
plt.legend(f, ['Actual','Pure Python','Numba CPU','PyOmp','Pyk CPU'])
plt.savefig('sflux.png', dpi=500, facecolor='w', edgecolor='k',orientation='portrait')  
