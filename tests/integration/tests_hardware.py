import numpy as np
import math
import mcdc_tnt
from timeit import default_timer as timer

def error(sim, bench):
    error = np.linalg.norm(sim - bench) / np.linalg.norm(bench)
    return(error)

if __name__ == '__main__':
    
    print()
    print('ATTENTION')
    print('Entering Hardware Test Suite')
    print('Ensure the proper conda enviorment is enabled')
    print('Test Schedule ([x] will run, [c] can run (must be manually set)):')
    print('     -[x] pure python')
    print('     -[x] numba cpu')
    print('     -[ ] numba gpu')
    print('     -[c] pykokkos cpu')
    print('     -[ ] pykokkos gpu')
    print('     -[c] pyomp cpu')
    print('This can take a while, recomended Pytest is not used')
    print()
    start_o = timer()
    
    print('Entering Pure Python')    
    input_file = 'tc_1_pp.yaml'
    output_file = 'pp.out'
    start = timer()
    mcdc_tnt.run(input_file, output_file, None)
    end = timer()
    time_pp = end-start
    
    print()
    print('Entering Numba CPU')   

    input_file = 'tc_1_numba_cpu.yaml'
    output_file = 'numba_cpu.out'
    start = timer()
    mcdc_tnt.run(input_file, output_file, None)
    end = timer()
    time_nbc = end-start
    
    #print()
    #print('Entering Numba GPU')   
    
    #input_file = 'tc_1_numba_gpu.yaml'
    #output_file = 'numba_gpu.out'
    #start = timer()
    #mcdc_tnt.run(input_file, output_file)
    #end = timer()
    #time_nbg = end-start
    
    #print()
    #print('Entering PyKokkos CPU')   
    
    #input_file = 'tc_1_pyk_cpu.yaml'
    #output_file = 'pyk_cpu.out'
    #start = timer()
    #mcdc_tnt.run(input_file, output_file)
    #end = timer()
    #time_pykc = end-start
    
    end_o = timer()
    
    sf_actual = np.loadtxt('anwser.pout', comments='#', delimiter=',', skiprows=2)
    
    sf_pp = np.loadtxt('pp.out', comments='#', delimiter=',', skiprows=2)
    sf_nbc = np.loadtxt('numba_cpu.out', comments='#', delimiter=',', skiprows=2)
    #sf_nbg = np.loadtxt('numba_gpu.out', comments='#', delimiter=',', skiprows=2) 
    #sf_pykc = np.loadtxt('pyk_cpu.out', comments='#', delimiter=',', skiprows=2) 
    
    
    assert(np.allclose(sf_actual[:,2], sf_pp[:,2], rtol=1e-01))
    assert(np.allclose(sf_actual[:,2], sf_nbc[:,2], rtol=1e-01))
    #assert(np.allclose(sf_actual[:,2], sf_nbg[:,2]))
    #assert(np.allclose(sf_actual[:,2], sf_pykc[:,2], rtol=1e-01))
    
    print()
    print('Test Complete and all Passed!')
    print('Total time to completion:')
    print('     -pure python.....{0}'.format(time_pp))
    print('     -numba cpu.......{0}'.format(time_nbc))
    #print('     -numba gpu.......{0}'.format(time_nbg))
    #print('     -pykokkos cpu....{0}'.format(time_pykc))
    print()
    print('     -total...........{0}'.format(end_o-start_o))
    print()
    print('Produced Errors Between Soultions')
    print('     -pure python............{0}'.format(error(sf_actual, sf_pp)))
    print('     -numba threading........{0}'.format(error(sf_actual, sf_nbc)))
    #print('     -numba pyomp............{0}'.format(error(sf_actual, sf_pyomp)))
    #print('     -pyk ompenmp............{0}'.format(error(sf_actual, sf_pykc)))
    print()
    
    import matplotlib.pyplot as plt
    plt.figure(1)
    f = plt.plot(sf_actual[:,1], sf_actual[:,2], '-b',
             sf_pp[:,1], sf_pp[:,2], '-r',
             sf_nbc[:,1], sf_nbc[:,2], 'g-')

    plt.title("Scalar Flux")
    plt.ylabel("$\phi [cm^{-2}s^{-1}]$")
    plt.xlabel("x [cm]")
    plt.legend(f, ['Actual','Pure Python','Numba CPU','Pyk CPU'])
    plt.savefig('sflux.png', dpi=500, facecolor='w', edgecolor='k',orientation='portrait')  
    print('Flux figure printed to sflux.png')
    print()
    #sf_pykc[:,1], sf_pykc[:,2], 'k-')
    print()
    
    
