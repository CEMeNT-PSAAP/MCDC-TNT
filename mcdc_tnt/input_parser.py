import numpy as np
import yaml

def SimulationSetup(input_file):
    """
    Sets up dictionaries and mesh for transport

    Parameters
    ----------
    input_file : <input_file_name>.yaml
        Name of the input file to be ran.

    Returns
    -------
    Inital PSV's for use in transport.

    """
    
    with open(input_file,'r') as f:
        inputs = yaml.safe_load(f)    
    #===============================================================================
    # Simulation settings (input deck)
    #===============================================================================
    
    seed = int(inputs['rng seed'])
    num_part = int(np.float((inputs['number of particles']))) #number of particles to start
    particle_speed = np.float(inputs['particle speed']) #particle speed
    
    # generations = 1
    nu_new_neutrons = int(inputs['neutrons per fission']) #neutrons/fission
    isotropic = inputs['isotropic'] #isotropic
    
    #===============================================================================
    # Test case 1: Single Reigon
    #===============================================================================
    
    Length_slab = np.float(inputs['length of slab'])
    surface_distances = np.array(inputs['surface locations'], dtype=float)
    
    mesh_cell_length = np.float(inputs['dx']) #dx
    N_mesh = int(Length_slab/mesh_cell_length)
    
    cap_xsec = np.float(inputs['capture cross section']) #capture crossection
    scat_xsec = np.float(inputs['scatter cross section'])  #scattering crossection
    fis_xsec = np.float(inputs['fission cross section'])  #fission crossection
    
    hardware_target = inputs['hardware target']
    
    sim_name = inputs['name']
    
    make_out = inputs['file output']
    
    #abs_xsec = cap_xsec+fis_xsec #absorption crossection
    total_xsec = cap_xsec + scat_xsec + fis_xsec #total crossection
    
    #assemble mesh
    amm = inputs['assemble mesh']
    
    p_warmup = inputs['print warmup times']
    plot_flux = inputs['flux plot']
    plot_error = inputs['error plot']
    
    
    if (amm == True):
        #establishing mesh
        mesh_scat_xsec = np.zeros(N_mesh, dtype=float)
        mesh_cap_xsec = np.zeros(N_mesh, dtype=float)
        mesh_fis_xsec = np.zeros(N_mesh, dtype=float)
        mesh_total_xsec = np.zeros(N_mesh, dtype=float)

        for cell in range(N_mesh):
            mesh_scat_xsec[cell] = scat_xsec
            mesh_cap_xsec[cell] = cap_xsec
            mesh_fis_xsec[cell] = fis_xsec
            mesh_total_xsec[cell] = total_xsec
    else:
        print('import mesh data from file')
        #import mesh from file
    
    #assemble formatted dicts for simplified i/o
    comp_parms = {'seed': seed,
                  'hard_targ': hardware_target,
                  'p_warmup': p_warmup,
                  'plot flux': plot_flux,
                  'plot error': plot_error,
                  'sim name': sim_name,
                  'output file': make_out}
                  
    sim_perams = {'num': num_part,
                  'L_slab': Length_slab,
                  'dx': mesh_cell_length,
                  'N_mesh': N_mesh,
                  'nu': nu_new_neutrons,
                  'iso': isotropic,
                  'part_speed': particle_speed}
                   
    
    #===============================================================================
    # Test case 2: Three Region
    #===============================================================================
    
    # # index refers to region
    # cap_xsec = np.array([1/3, 1/3, 1/3]) #capture crossection
    # scat_xsec = np.array([2/3, 1/3, 2/3]) #scattering crossection
    # fis_xsec = np.array([0,1/3,0]) #fission crossection
    
    # #abs_xsec = cap_xsec+fis_xsec #absorption crossection
    # total_xsec = np.array([1,1,1]) #total crossection
    
    # Length_slab = 5
    # surface_distances = np.array([0,2,4,Length_slab], dtype=np.float32)
    # mesh_cell_length = 0.2 #dx
    # N_mesh = int(Lenght_slab/mesh_cell_length)
    
    # #establishing mesh
    # mesh_scat_xsec = np.zeros(N_mesh, dtype=np.float32)
    # mesh_cap_xsec = np.zeros(N_mesh, dtype=np.float32)
    # mesh_fis_xsec = np.zeros(N_mesh, dtype=np.float32)
    # mesh_total_xsec = np.zeros(N_mesh, dtype=np.float32)
    
    # for i in range(len(surface_distances)-1):
    #     for cell in range(int(surface_distances[i]/mesh_cell_length), int(surface_distances[i+1]/mesh_cell_length)):
    #         mesh_scat_xsec[cell] = scat_xsec[i]
    #         mesh_cap_xsec[cell] = cap_xsec[i]
    #         mesh_fis_xsec[cell] = fis_xsec[i]
    #         mesh_total_xsec[cell] = total_xsec[i]
        
    
    # L = [0,2,4,5] #coordinants of boundaries
    # generation_region = 1
    # regions = 3
    
    return(comp_parms, sim_perams, mesh_cap_xsec, mesh_scat_xsec, mesh_fis_xsec, mesh_total_xsec, surface_distances)
