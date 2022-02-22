"""
Created on Tue Jan 25 11:19:50 2022

@author: jack
"""

import generations
from input_parser import SimulationSetup
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse


def run():
    parser = argparse.ArgumentParser(description='Main file to run MC/DC-TNT')
    parser.add_argument('-i', '--input', required=True,
                        help='input file in a .yaml format (see InputDeck.py)')
    parser.add_argument('-o', '--output', required=False,
                        help='output file, if none then output.txt')
                        
    args = parser.parse_args(sys.argv[1:])

    input_file = args.input
    output_file = args.output

    [comp_parms, sim_perams, mesh_cap_xsec, mesh_scat_xsec, mesh_fis_xsec, mesh_total_xsec, surface_distances] = SimulationSetup(input_file)

    [scalar_flux, standard_deviation_flux] = generations.Generations(comp_parms, sim_perams, mesh_cap_xsec, mesh_scat_xsec, mesh_fis_xsec, mesh_total_xsec, surface_distances)

    if (output_file == None):
        print("Simulation Complete, no outputs")

    x_mesh = np.linspace(0,1,len(scalar_flux))

    scalar_flux /= np.max(scalar_flux)
    standard_deviation_flux /= np.max(standard_deviation_flux)
    
    plt.figure(1)
    plt.plot(x_mesh, scalar_flux, '-b', x_mesh, standard_deviation_flux, '--k')
    plt.title("Scalar Flux: ", )
    plt.ylabel("$\phi [cm^{-2}s^{-1}]$")
    plt.xlabel("x [cm]")
    plt.show()


if __name__ == "__main__":
    run()
    
# plt.figure(2)
# plt.plot(x_mesh, standard_deviation_flux, 'b-')
# plt.title('Standard Deviation')
# plt.ylabel('$\sigma$')
# plt.xlabel('cell')



# mu_0 = 0.87     #cosine of the average scattering angle

# Sig_t = 1             #total macroscopic x-section [1/cm]
# Sig_tr = (Sig_t - mu_0*1/3)    #macroscopic transport x-section [1/cm]

# z0 = 0.7109*(1/Sig_tr)           #vacumme extension [cm]; D&H eqn. 5-7
# a_bar = L+z0
# x_mesh_bar = np.linspace(-surface_distances[len(surface_distances)-1]/2,surface_distances[len(surface_distances)-1]/2, N_mesh)
# scalar_flux_buckling = np.cos(np.pi*x_mesh_bar/a_bar)

# plt.figure(3)
# plt.plot(x_mesh, scalar_flux, '-b', x_mesh, scalar_flux_buckling, 'k--*')
# plt.title("Scalar Flux")
# plt.xlabel("$\phi [cm^{-2}s^{-1}]$")
# plt.ylabel("x [cm]")


# print("")
# print("leak left: {0}".format(trans_lhs/init_particle))
# print("")
# print("leak right: {0}".format(trans_rhs/init_particle))
# print("")
# print('')
# print("********************END SIMULATION********************")
# print('')
