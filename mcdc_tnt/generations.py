import numpy as np
from timeit import default_timer as timer



#import numba_kernels.cpu as kernels
#import pyk_kernels.ad_o as kernels
#===============================================================================
# Simulation Setup
#===============================================================================



#===============================================================================
# EVENT 0 : Sample particle sources
#===============================================================================

#@nb.jit(nopython=True)
def Generations(comp_parms, sim_perams, mesh_cap_xsec, mesh_scat_xsec, mesh_fis_xsec, mesh_total_xsec, surface_distances):
    """
    Runs a generation of transport. Each one is launched in complete isolation of
    another

    Parameters
    ----------
    comp_parms : Python Dict
        variables important for the computation (e.g. number of cores).
    sim_perams : Python Dict
        variables for simulation (e.g. num particles).
    mesh_cap_xsec : vector double
        capture x-sections for every mesh cell.
    mesh_scat_xsec : vector double
        scattering x-sections for every mesh cell.
    mesh_fis_xsec : vector double
        fission x-sections for every mesh cell.
    mesh_total_xsec : vector double
        total x-sections for every mesh cell.
    surface_distances : vector double
        location of material interfaces defining "regions".

    Returns
    -------
    scalar flux and assocated errors.

    """
    
    if comp_parms['hard_targ'] == 'pp':
        import mcdc_tnt.pp_kernels as kernels
        
    elif comp_parms['hard_targ'] == 'nb_cpu':
        print('loading cpu')
        import mcdc_tnt.numba_kernels.cpu as kernels
        #from mcdc_tnt.numba_kernels.warmup import WarmUp
        #WarmUp(comp_parms['p_warmup']) #warmup kernels
        
    elif comp_parms['hard_targ'] == 'nb_gpu':
        print('loading gpu')
        import mcdc_tnt.numba_kernels.gpu as kernels
        #from mcdc_tnt.numba_kernels.warmup import WarmUp
        #WarmUp(comp_parms['p_warmup']) #warmup kernels
    
    N_mesh = sim_perams['N_mesh']
    nu_new_neutrons = sim_perams['nu']
    num_part = sim_perams['num']
    dx = sim_perams['dx']
    particle_speed = sim_perams['part_speed']
    dat_type = comp_parms['data type']
    dt = sim_perams['dt']
    max_time = sim_perams['max time']
    N_time = sim_perams['N_time']
    
    #===============================================================================
    # Initial setups
    #===============================================================================
    
    # Initialize RNG
    np.random.seed(comp_parms['seed'])
    
    init_particle = num_part
    meshwise_fission_pdf = np.zeros(N_mesh, dtype=dat_type)
    
    total_mesh_fission_xsec = sum(mesh_fis_xsec)
    for cell in range(N_mesh):
        meshwise_fission_pdf[cell] = mesh_fis_xsec[cell]/total_mesh_fission_xsec
        
        mesh_cap_xsec[cell] = mesh_cap_xsec[cell] / mesh_total_xsec[cell]
        mesh_scat_xsec[cell] = mesh_scat_xsec[cell] / mesh_total_xsec[cell]
        mesh_fis_xsec[cell] = mesh_fis_xsec[cell] / mesh_total_xsec[cell]
        
    meshwise_fission_pdf /= sum(meshwise_fission_pdf)
    
    # last Ntime itter is for all post times in the particle generation scheem
    
    mesh_dist_traveled = np.zeros([N_mesh, N_time], dtype=dat_type)
    mesh_dist_traveled_squared = np.zeros([N_mesh, N_time], dtype=dat_type)
    
    
    #===============================================================================
    # Allocate particle phase space
    #===============================================================================
    
    phase_parts = int(1.5*num_part) #see note about data storage
    
    # Position
    p_pos_x = np.zeros(phase_parts, dtype=dat_type)
    p_pos_y = np.zeros(phase_parts, dtype=dat_type)
    p_pos_z = np.zeros(phase_parts, dtype=dat_type)
    
    # Direction
    p_dir_x = np.zeros(phase_parts, dtype=dat_type)
    p_dir_y = np.zeros(phase_parts, dtype=dat_type)
    p_dir_z = np.zeros(phase_parts, dtype=dat_type)
    
    # Speed
    p_speed = np.zeros(phase_parts, dtype=dat_type)
    
    # Time
    p_time = np.zeros(phase_parts, dtype=dat_type)
    p_time_cell = np.zeros(phase_parts, dtype=np.int32)
    
    # Region
    p_mesh_cell = np.zeros(phase_parts, dtype=np.int32)
    #print(p_mesh_cell.dtype)
    # Flags
    p_alive = np.full(phase_parts, False, dtype=np.int32)
    
    scatter_event_index = np.zeros(phase_parts, dtype=np.int32)
    capture_event_index = np.zeros(phase_parts, dtype=np.int32)
    fission_event_index = np.zeros(phase_parts, dtype=np.int32)
    
    
    [p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, 
    p_alive] = kernels.SourceParticles(p_pos_x, p_pos_y,
                                                      p_pos_z, p_mesh_cell, dx,
                                                      p_dir_y, p_dir_z, p_dir_x,
                                                      p_speed, p_time, p_alive,
                                                      num_part, meshwise_fission_pdf,
                                                      particle_speed, sim_perams['iso'])
    
    
    total_size = p_pos_x.nbytes + p_pos_y.nbytes + p_pos_z.nbytes + p_dir_z.nbytes + p_dir_x.nbytes + p_dir_y.nbytes + p_mesh_cell.nbytes + p_speed.nbytes + p_time.nbytes + p_alive.nbytes + mesh_total_xsec.nbytes + mesh_dist_traveled.nbytes + mesh_dist_traveled_squared.nbytes
    
    print('Total size of problem to be sent to GPU: {0} GB'.format(total_size/1e9))
    
    #===============================================================================
    # Generation Loop
    #===============================================================================
    trans = 0
    g = 1
    alive = num_part
    trans_lhs = 0
    trans_rhs = 0
    tally_time = 0
    switch_flag = 0
    while alive > 100:
        print("")
        print("===============================================================================")
        print("                             Event Cycle {0}".format(g))
        print("===============================================================================")
        print("particles alive at start of event cycle {0}".format(num_part))
        
        start_o = timer()
        
        # print("max index {0}".format(num_part))
        #===============================================================================
        # EVENT 1 : Advance
        #===============================================================================
        killed = 0
        alive_cycle_start = num_part
        
        start = timer()
        if (comp_parms['hard_targ'] == 'nb_gpu') and (num_part < 50000) and (switch_flag == 0):
            print('Switching to CPU')
            import mcdc_tnt.numba_kernels.cpu as kernels
            switch_flag = 1
        
        #print(mesh_dist_traveled_squared.size)
        #print(mesh_dist_traveled.size)
        
        [p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, mesh_dist_traveled, mesh_dist_traveled_squared] = kernels.Advance(
                p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, dt, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_time_cell,
                num_part, mesh_total_xsec, mesh_dist_traveled, mesh_dist_traveled_squared, surface_distances[len(surface_distances)-1], max_time)
        
        
        #print(mesh_dist_traveled)
        
        end = timer()
        print('Advance time: {0}'.format(end-start))
        #===============================================================================
        # EVENT 2a : Still in problem Spatially
        #===============================================================================
        [p_alive, tally_left_t, tally_right_t] = kernels.StillInSpace(p_pos_x, surface_distances, p_alive, num_part)
        
        trans_lhs += tally_left_t
        trans_rhs += tally_right_t
        
        #===============================================================================
        # EVENT 2b : Still in problem Temporally
        #===============================================================================
        
        #[p_alive, tally_time_t] = kernels.StillInTime(p_time, max_time, p_alive, num_part)
        
        #tally_time += tally_time_t
        
        #===============================================================================
        # EVENT 3 : Sample event
        #===============================================================================
        
        rands = np.random.random(num_part)
        
        [scatter_event_index, scat_count, capture_event_index, cap_count, fission_event_index, fis_count] = kernels.SampleEvent(
                p_mesh_cell, p_alive, mesh_cap_xsec, mesh_scat_xsec, mesh_fis_xsec, scatter_event_index,
                capture_event_index, fission_event_index, num_part, nu_new_neutrons, rands)
       
        
        fissions_to_add = (fis_count)*nu_new_neutrons
        
        killed += cap_count+fis_count
        
        
        #===============================================================================
        # EVENT 3 : Scatter
        #===============================================================================
        
        rands = np.random.random(scat_count * 2) #exact number of rands known
        
        [p_dir_x, p_dir_y, p_dir_z] = kernels.Scatter(scatter_event_index, scat_count, p_dir_x, p_dir_y, p_dir_z, rands)
        
        
        #===============================================================================
        # EVENT 4: Generate fission particles
        #===============================================================================
        # print("")
        # print("max index {0}".format(num_part))
        # print("")
        
        rands = np.random.random(fis_count * nu_new_neutrons * 2) #exact number of rands known
        #2 is for number reqired per new neutron
        
        [p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, 
         p_time, p_alive, particles_added_fission] = kernels.FissionsAdd(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, 
                                                  p_dir_y, p_dir_z, p_dir_x, p_speed, 
                                                  p_time, p_alive, fis_count, nu_new_neutrons, 
                                                  fission_event_index, num_part, particle_speed, rands)
    
        num_part += particles_added_fission
        # print("")
        # print("max index {0}".format(num_part))
                                                  
        #===============================================================================
        # Criticality & Output (not really an event)
        #===============================================================================
        
        # criticality = fissions_to_add/killed
        # print("k = {0} (birth/death)".format(criticality))
                
        # alive_now =0
        # for i in range(num_part):
        #     if p_alive[i] == True:
        #         alive_now +=1
        # print("k = {0} (pop now/pop last)".format(alive_now/alive_last))
        
        
        #===============================================================================
        # Event 5: Purge the dead
        #===============================================================================
        
        [p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, 
         p_time, p_alive, kept] = kernels.BringOutYourDead(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, 
                                                   p_dir_y, p_dir_z, p_dir_x, p_speed, 
                                                   p_time, p_alive, num_part)
                                                   
        num_part = kept
        alive = num_part
                          
        # print("max index {0}".format(num_part))
        # print("")
        
        # print(max(p_mesh_cell[0:num_part]))
        g+=1
        
        end_o = timer()
        print('Overall time to completion: {0}'.format(end_o-start_o))
    #===============================================================================
    # Step Output
    #===============================================================================
    
    
    mesh_dist_traveled /= init_particle
    mesh_dist_traveled_squared /= init_particle
    standard_deviation_flux = ((mesh_dist_traveled_squared - mesh_dist_traveled**2)/(init_particle-1))
    standard_deviation_flux = np.sqrt(standard_deviation_flux/(init_particle))
    
    #x_mesh = np.linspace(0,surface_distances[len(surface_distances)-1], N_mesh)
    scalar_flux = mesh_dist_traveled/dx
    #scalar_flux/=max(scalar_flux)
    
    return(scalar_flux, standard_deviation_flux)
    
    # # the sum of all debits from functional operations should be the number of
    # #currently alive particles
    # account = alive_cycle_start + particles_added_fission - fis_count - cap_count - tally_left_t - tally_right_t
    # if account != num_part:
    #     print("ERROR PARTICLES HAVE BEEN UNACCOUNTED FOR")
    
    # print("{0} particles are produced from {1} fission events".format(fissions_to_add, fis_count))
    # print("particles captured:  {0}".format(cap_count))
    # print("particles scattered: {0}".format(scat_count))
    # print("particles leaving left:    {0}".format(tally_left_t))
    # print("particles leaving right:   {0}".format(tally_right_t))
    # print("total particles now alive and stored: {0}".format(num_part))
    
    # # alive_last = alive_now
    
def mem_note():
    total_size = p_pos_x.nbytes + p_pos_y.nbytes + p_pos_z.nbytes + p_dir_z.nbytes + p_dir_x.nbytes + p_dir_y.nbytes + p_mesh_cell.nbytes + p_speed.nbytes + p_time.nbytes + p_alive.nbytes + mesh_total_xsec.nbytes + mesh_dist_traveled.nbytes + mesh_dist_traveled_squared.nbytes
    
    print('Total size of problem to be sent to GPU: {0} GB'.format(total_size/1e9))

if __name__ == '__main__':
    x=0



