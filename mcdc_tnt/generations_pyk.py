import numpy as np
import pykokkos as pk
import mcdc_tnt.pyk_kernels.all as kernels

#===============================================================================
# Simulation Setup
#===============================================================================



#===============================================================================
# EVENT 0 : Sample particle source
#===============================================================================

def Generations(comp_parms, sim_perams, mesh_cap_xsec_np, mesh_scat_xsec_np, mesh_fis_xsec_np, mesh_total_xsec_np, surface_distances_np):
    """
    Runs a generation of transport. Eachone is launched in complete isolation of
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
    #import_case(comp_parms['hard_targ'])
    
    #===============================================================================
    # Pykokkos Setup
    #===============================================================================
    pk.set_default_space(pk.ExecutionSpace.OpenMP)
    
    N_mesh = sim_perams['N_mesh']
    nu_new_neutrons = sim_perams['nu']
    num_part = sim_perams['num']
    dx = sim_perams['dx']
    particle_speed = sim_perams['part_speed']
    
    
    #===============================================================================
    # Initial setups
    #===============================================================================
    
    # Initialize RNG
    np.random.seed(comp_parms['seed'])
    
    
    init_particle = num_part
    
    
    total_mesh_fission_xsec = sum(mesh_fis_xsec_np)
    
    meshwise_fission_pdf_np = mesh_fis_xsec_np/total_mesh_fission_xsec
    meshwise_fission_pdf = pk.from_numpy(meshwise_fission_pdf_np)
    
    
    mesh_cap_xsec = pk.from_numpy(mesh_cap_xsec_np)
    mesh_scat_xsec = pk.from_numpy(mesh_scat_xsec_np)
    mesh_fis_xsec = pk.from_numpy(mesh_fis_xsec_np)
    mesh_total_xsec = pk.from_numpy(mesh_total_xsec_np)
    
    meshwise_fission_pdf_np /= sum(meshwise_fission_pdf_np)
    meshwise_fission_pdf = pk.from_numpy(meshwise_fission_pdf_np)
    
    mesh_dist_traveled_np = np.zeros(N_mesh, dtype=float)
    mesh_dist_traveled = pk.from_numpy(mesh_dist_traveled_np)
    
    mesh_dist_traveled_squared_np = np.zeros(N_mesh, dtype=float)
    mesh_dist_traveled_squared = pk.from_numpy(mesh_dist_traveled_squared_np)
    
    
    
    #===============================================================================
    # Allocate particle phase space
    #===============================================================================
    
    phase_parts = 5*num_part #see note about data storage
    
    # Position
    p_pos_x_np = np.zeros(phase_parts, dtype=float)
    p_pos_y_np = np.zeros(phase_parts, dtype=float)
    p_pos_z_np = np.zeros(phase_parts, dtype=float)
    
    p_pos_x = pk.from_numpy(p_pos_x_np)
    p_pos_y = pk.from_numpy(p_pos_y_np)
    p_pos_z = pk.from_numpy(p_pos_z_np)
    
    # Direction
    p_dir_x_np = np.zeros(phase_parts, dtype=float)
    p_dir_y_np = np.zeros(phase_parts, dtype=float)
    p_dir_z_np = np.zeros(phase_parts, dtype=float)
    
    p_dir_x = pk.from_numpy(p_dir_x_np)
    p_dir_y = pk.from_numpy(p_dir_y_np)
    p_dir_z = pk.from_numpy(p_dir_z_np)
    
    # Speed
    p_speed_np = np.zeros(phase_parts, dtype=float)
    p_speed = pk.from_numpy(p_speed_np)
    
    # Time
    p_time_np = np.zeros(phase_parts, dtype=float)
    p_time = pk.from_numpy(p_time_np)
    
    # Region
    p_mesh_cell_np = np.zeros(phase_parts, dtype=np.int32)
    p_mesh_cell = pk.from_numpy(p_mesh_cell_np)
    
    # Flags
    p_alive_np = np.full(phase_parts, False, dtype=np.int32)
    p_alive = pk.from_numpy(p_alive_np)
    
    #mesh_particle_index = np.zeros([N_mesh, phase_parts], dtype=np.uint8)
    
    scatter_event_index_np = np.zeros(phase_parts, dtype=np.int32)
    capture_event_index_np = np.zeros(phase_parts, dtype=np.int32)
    fission_event_index_np = np.zeros(phase_parts, dtype=np.int32)
    
    scatter_event_index = pk.from_numpy(scatter_event_index_np)
    capture_event_index = pk.from_numpy(capture_event_index_np)
    fission_event_index = pk.from_numpy(fission_event_index_np)
    
    surface_distances = pk.from_numpy(surface_distances_np)
    print(surface_distances.dtype)
    
    rands_np = np.random.random([num_part*4])
    rands = pk.from_numpy(rands_np)
    
    #print(p_pos_x.dtype)
    #print(p_pos_y.dtype)
    #print(p_pos_z.dtype)
    #print(p_mesh_cell.dtype)
    #print(p_dir_y.dtype)
    #print(p_dir_z.dtype)
    #print(p_dir_x.dtype)
    #print(p_speed.dtype)
    #print(p_time.dtype)
    #print(p_alive.dtype)
    #print(meshwise_fission_pdf.dtype)
    #print(rands.dtype)
    
    print('Entering Source!')
    timer = pk.Timer()
    
    pk.execute(pk.ExecutionSpace.Default, 
        kernels.SourceParticles(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive, num_part, particle_speed, meshwise_fission_pdf, rands))
          #                      p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive, num_parts, particle_speed, meshwise_fission_pdf, rands
    res = timer.seconds()
    print('Source function time {0}'.format(res)) 
    #===============================================================================
    # Generation Loop
    #===============================================================================
    trans = 0
    g = 0
    alive = num_part
    trans_lhs = 0
    trans_rhs = 0
    
    #pk view to export needed integer values form a function
    clever_out: pk.View1D[int] = pk.View([10], pk.int32)
    
    while alive > 0:
        print("")
        print("===============================================================================")
        print("                             Event Cycle {0}".format(g))
        print("===============================================================================")
        print("particles alive at start of event cycle {0}".format(num_part))
        
        timer_r = pk.Timer()
        # print("max index {0}".format(num_part))
        #===============================================================================
        # EVENT 1 : Advance
        #===============================================================================
        killed = 0
        alive_cycle_start = num_part
        
        print('Entering Advance!')
        timer = pk.Timer()
        
        kernels.Advance(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, num_part, mesh_total_xsec, mesh_dist_traveled, mesh_dist_traveled_squared, surface_distances[len(surface_distances)-1])
        
        res = timer.seconds()
        print('Advance function time {0}'.format(res))
        timer = pk.Timer()
        
        #print(sum(p_alive[0:num_part]))  
        #===============================================================================
        # EVENT 2 : Still in problem
        #===============================================================================
        print('Entering StillIn!')
        pk.execute(pk.ExecutionSpace.Default, kernels.StillIn(p_pos_x, surface_distances, p_alive, num_part, clever_out))
        
        res = timer.seconds()
        print('Still in function time {0}'.format(res))
        
        
        trans_lhs += clever_out[0]
        trans_rhs += clever_out[1]
        
        #print(sum(p_alive[0:num_part]))  
        #===============================================================================
        # EVENT 3 : Sample event
        #===============================================================================
        
        rands_np = np.random.random(num_part)
        rands = pk.from_numpy(rands_np)
        
        print('Entering Sample!')
        #print(p_mesh_cell.dtype)
        #print(p_alive.dtype)
        #print(mesh_cap_xsec.dtype)
        #print(p_mesh_cell.dtype)
        #print(mesh_scat_xsec.dtype)
        #print(mesh_fis_xsec.dtype)
        #print(scatter_event_index.dtype)
        #print(capture_event_index.dtype)
        #print(clever_out.dtype)
        #print(p_alive.dtype)
        #print(meshwise_fission_pdf.dtype)
        #print(rands.dtype)
        timer = pk.Timer()
        
        pk.execute(pk.ExecutionSpace.Default, kernels.SampleEvent(p_mesh_cell, p_alive, mesh_cap_xsec, mesh_scat_xsec, mesh_fis_xsec, scatter_event_index,
                                capture_event_index, fission_event_index, num_part, nu_new_neutrons, rands, clever_out))
        
        res = timer.seconds()
        print('Sample event in function time {0}'.format(res))
        #print(sum(p_alive[0:num_part]))
        scat_count = clever_out[0]
        cap_count = clever_out[1]
        fis_count = clever_out[2] 
       
        
        fissions_to_add = (fis_count)*nu_new_neutrons
        
        killed += cap_count+fis_count
        
        
        #===============================================================================
        # EVENT 3 : Scatter
        #===============================================================================
        
        rands_np = np.random.random(scat_count * 2) #exact number of rands known
        rands = pk.from_numpy(rands_np)
        
        timer = pk.Timer()
        
        print('Entering Scatter!')
        pk.execute(pk.ExecutionSpace.Default, kernels.Scatter(scatter_event_index, scat_count, p_dir_x, p_dir_y, p_dir_z, rands))
        
        res = timer.seconds()
        print('Scatter function time {0}'.format(res))
        
        
        #===============================================================================
        # EVENT 4: Generate fission particles
        #===============================================================================
        # print("")
        # print("max index {0}".format(num_part))
        # print("")
        
        rands_np = np.random.random(fis_count * nu_new_neutrons * 2) #exact number of rands known
        rands = pk.from_numpy(rands_np)
        #2 is for number reqired per new neutron
        timer = pk.Timer()
        
        print('Entering Fissions!')
        pk.execute(pk.ExecutionSpace.Default, kernels.FissionsAdd(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, 
                                                  p_dir_y, p_dir_z, p_dir_x, 
                                                  p_time, p_alive, p_speed, fis_count, nu_new_neutrons, 
                                                  fission_event_index, num_part, particle_speed, rands, clever_out))
        res = timer.seconds()
        print('Fissions function time {0}'.format(res))
        #print(sum(p_alive[0:num_part]))  
        num_part += clever_out[0]
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
        print('Entering PURGE!')
        timer = pk.Timer()
        pk.execute(pk.ExecutionSpace.Default, kernels.BringOutYourDead(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, 
                                                   p_dir_y, p_dir_z, p_dir_x, p_speed, 
                                                   p_time, p_alive, num_part, clever_out))
        #print(sum(p_alive[0:num_part]))         
        res = timer.seconds()
        print('CleanUp function time {0}'.format(res))
        num_part = clever_out[0]
        alive = num_part
                          
        # print("max index {0}".format(num_part))mesh_fis_xsec
        # print("")
        
        # print(max(p_mesh_cell[0:num_part]))
        g+=1
        
        res_r = timer_r.seconds()
        print('Cycle function time {0}'.format(res_r))
    #===============================================================================
    # Step Output
    #===============================================================================
    
    #get back from pyk views
    for i in range(N_mesh):
        mesh_dist_traveled_np[i] = mesh_dist_traveled[i]
        mesh_dist_traveled_squared_np[i] = mesh_dist_traveled_squared[i]
    
    
    mesh_dist_traveled_np /= init_particle
    mesh_dist_traveled_squared_np /= init_particle
    standard_deviation_flux = ((mesh_dist_traveled_squared_np - mesh_dist_traveled_np**2)/(init_particle-1))
    standard_deviation_flux = np.sqrt(standard_deviation_flux/(init_particle))
    
    x_mesh = np.linspace(0,surface_distances[len(surface_distances)-1], N_mesh)
    scalar_flux = mesh_dist_traveled_np/dx
    scalar_flux/=max(scalar_flux)
    
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
    


if __name__ == '__main__':
    x=0



