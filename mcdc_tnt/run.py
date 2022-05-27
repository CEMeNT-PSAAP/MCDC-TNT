#from .input_parser import SimulationSetup
import numpy as np
import sys
import argparse
import mcdc_tnt
import matplotlib.pyplot as plt

def flatLinePlot(x, y, lab):
    for i in range(y.size):
        xx = x[i:i+2]
        yy = [y[i], y[i]]
        plt.plot(xx, yy, label=lab)

def run(input_file, output_file=None, hard_targ=None):
    """
    main function to run a single generation and plot the output

    Returns
    -------
    Plots and output tables if requested.

    """
    
    with open("title_print.txt", "r", encoding="utf-8") as file:
        for line in file:
            print(line.strip())
    
    [comp_parms, sim_perams, mesh_cap_xsec, mesh_scat_xsec, mesh_fis_xsec, mesh_total_xsec, surface_distances] = mcdc_tnt.SimulationSetup(input_file)
    
    if hard_targ != None:
         comp_parms['hard_targ'] = hard_targ
    
    
    if comp_parms['hard_targ'] == 'pp':
        from mcdc_tnt.generations import Generations
        print('>>>Running Prue Python kernels (slow)')
    elif comp_parms['hard_targ'] == 'nb_cpu':
        from mcdc_tnt.generations import Generations
        print('>>>Running Numba CPU kernels')
    elif comp_parms['hard_targ'] == 'nb_gpu':
        from mcdc_tnt.generations import Generations
        print('>>>Running Numba GPU kernels (slow)')
    elif comp_parms['hard_targ'] == 'pyk_cpu':
        from mcdc_tnt.generations_pyk import Generations
        print('>>>Running PyKokkos CPU kernels')
        print('    ensure correct conda enviroment is loaded!')
    elif comp_parms['hard_targ'] == 'pyk_gpu':
        print('>>>Feature not yet implemented, running pyk cpu kerenels')
        from mcdc_tnt.generations_pyk import Generations
        print('>>>Running PyKokkos CPU kernels')
        print('    ensure correct conda enviroment is loaded!')
    else:
        print()
        print('>>FATAL ERROR: NO HARDWARE TARGET<<')
        print()
        return()
    print()
    
    [scalar_flux, standard_deviation_flux] = Generations(comp_parms, sim_perams, mesh_cap_xsec, mesh_scat_xsec, mesh_fis_xsec, mesh_total_xsec, surface_distances)
    print()
    print('Simulation complete')
    print()
    
    x_mesh = np.linspace(0,sim_perams['L_slab'],80)#len(scalar_flux))
    X = np.linspace(0, sim_perams['L_slab'], 80+1)#int(scalar_flux.size+1))
    
    #print(scalar_flux)
    #scalar_flux /= np.max(scalar_flux)
    np.set_printoptions(threshold=np.inf)   
    
    if comp_parms['output file'] == True:
        if (output_file == None):
           output_file = 'output.out'
        with open(output_file, 'w') as f:
            print(comp_parms['sim name'],'output file', file=f)
            for i in range(scalar_flux.shape[1]):
                print((scalar_flux[:,i]), file=f)
            #print('cell, center x, normalized scalar flux, associated error', file=f)
            #for i in range(len(scalar_flux)):
            #    print('{0},{1},{2},{3}'.format(i, x_mesh[i], scalar_flux[i], standard_deviation_flux[i]), file=f) 
        print('Output written to',output_file)
        print()
    else:
        print('No file outputs requested, Simulation Complete')
    
    '''
    if comp_parms['plot error'] == True:
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.plot(x_mesh, standard_deviation_flux, '-k')
        plt.title(["$σ^2$ ",comp_parms['sim name']])
        plt.ylabel("$σ^2$")
        plt.xlabel("x [cm]")
        plt.savefig('error.png', dpi=500, facecolor='w', edgecolor='k',orientation='portrait')
        print('Error figure printed to error.png')
        print()
        #np.sum(scalar_flux, axis=1)
        '''
    #print(scalar_flux)
    #print()
    #print()
    
    #print(scalar_flux[0,:])
    #print()
    #print()
    #print(scalar_flux[:,0])
    
    import matplotlib.pyplot as plt
    plt.figure(2)
    for i in range(scalar_flux.shape[0]):
        plt.plot(x_mesh, scalar_flux[i,:], label=i)
    plt.show()
    
    '''
    if comp_parms['plot flux'] == True:
        import matplotlib.pyplot as plt
        plt.figure(2)
        print(scalar_flux.shape)
        for i in range(scalar_flux.shape[1]):
            plt.plot(x_mesh, scalar_flux[:,i], label=i)
            #flatLinePlot(X, scalar_flux[:, i], i)
        #plt.ylim([0,2])
        plt.grid(True)
        plt.title(["Scalar Flux: ",comp_parms['sim name']])
        plt.ylabel("$\phi [cm^{-2}s^{-1}]$")
        plt.xlabel("x [cm]")
        plt.legend(loc='right')
        
        plt.savefig('sflux.pdf', dpi=500, facecolor='w', edgecolor='k',orientation='portrait')  
        print('Flux figure printed to sflux.png')
        print()
    '''
    #print(scalar_flux.shape)
    #print()
    
    
    #plt.figure(3)
    #plt.plot(x_mesh, scalar_flux[:, 0])
    #plt.show()
    
    #for i in range(scalar_flux.shape[1]):
    #    print(sum(scalar_flux[:, i]))
    #print()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main file to run MC/DC-TNT')
    parser.add_argument('-i', '--input', required=True,
                        help='input file in a .yaml format (see InputDeck.py)')
    parser.add_argument('-o', '--output', required=False,
                        help='output file, if none then output.txt')
    parser.add_argument('-t', '--target', required=False,
                        help='hardware target, if none then use one listed in input.yaml (pp = pure python, nb_cpu = numba cpu)')
    args = parser.parse_args(sys.argv[1:])

    input_file = args.input
    output_file = args.output
    hard_targ = args.target

    run(input_file, output_file, hard_targ)
