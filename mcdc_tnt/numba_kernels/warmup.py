import mcdc_tnt.numba_kernels.cpu as kernels
import numpy as np
from timeit import default_timer as timer

def WarmUp(print_q):
    
    N_mesh = 2
    nu_new_neutrons = 2
    num_part = 1
    dx = 1
    particle_speed = 1
    
    mesh_cap_xsec = np.array([1,1], dtype=float)
    mesh_scat_xsec =  np.array([1,1], dtype=float)
    mesh_fis_xsec =  np.array([1,1], dtype=float)
    mesh_total_xsec =  np.array([3,3], dtype=float)
    surface_distances =  np.array([0,2], dtype=float)
    
    #===============================================================================
    # Initial setups
    #===============================================================================
    
    # Initialize RNG
    np.random.seed(777)
    
    init_particle = num_part
    
    meshwise_fission_pdf = mesh_fis_xsec/sum(mesh_fis_xsec)
    mesh_dist_traveled = np.zeros(N_mesh, dtype=float)
    mesh_dist_traveled_squared = np.zeros(N_mesh, dtype=float)
    
    
    #===============================================================================
    # Allocate particle phase space
    #===============================================================================
    
    phase_parts = 5*num_part #see note about data storage
    
    # Position
    p_pos_x = np.zeros(phase_parts, dtype=float)
    p_pos_y = np.zeros(phase_parts, dtype=float)
    p_pos_z = np.zeros(phase_parts, dtype=float)
    
    # Direction
    p_dir_x = np.zeros(phase_parts, dtype=float)
    p_dir_y = np.zeros(phase_parts, dtype=float)
    p_dir_z = np.zeros(phase_parts, dtype=float)
    
    # Speed
    p_speed = np.zeros(phase_parts, dtype=float)
    
    # Time
    p_time = np.zeros(phase_parts, dtype=float)
    
    # Region
    p_mesh_cell = np.zeros(phase_parts, dtype=np.int32)
    #print(p_mesh_cell.dtype)
    # Flags
    p_alive = np.full(phase_parts, False, dtype=bool)
    
    #mesh_particle_index = np.zeros([N_mesh, phase_parts], dtype=np.uint8)
    
    
    scatter_event_index = np.zeros(phase_parts, dtype=int)
    capture_event_index = np.zeros(phase_parts, dtype=int)
    fission_event_index = np.zeros(phase_parts, dtype=int)
    
    rands = np.random.random(phase_parts)
    
    start_o = timer()
    
    start = timer()
    [p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, 
    p_alive] = kernels.SourceParticles(p_pos_x, p_pos_y,
                                                      p_pos_z, p_mesh_cell, dx,
                                                      p_dir_y, p_dir_z, p_dir_x,
                                                      p_speed, p_time, p_alive,
                                                      num_part, meshwise_fission_pdf,
                                                      particle_speed, True)
    end = timer()
    time_source = end-start
    
    start = timer()
    [p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, mesh_dist_traveled, mesh_dist_traveled_squared] = kernels.Advance(
                p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time,
                num_part, mesh_total_xsec, mesh_dist_traveled, mesh_dist_traveled_squared, surface_distances[-1])
                
    end = timer()
    time_ad = end-start
    start = timer()
    
    [scatter_event_index, scat_count, capture_event_index, cap_count, fission_event_index, fis_count] = kernels.SampleEvent(
                p_mesh_cell, p_alive, mesh_cap_xsec, mesh_scat_xsec, mesh_fis_xsec, scatter_event_index,
                capture_event_index, fission_event_index, num_part, nu_new_neutrons, rands)
    
    end = timer()
    time_sample = end-start
    start = timer()
    
    [p_alive, tally_left_t, tally_right_t] = kernels.StillIn(p_pos_x, surface_distances, p_alive, num_part)
    
    end = timer()
    time_stillin = end-start
    start = timer()
    
    scat_count = 0 
    [p_dir_x, p_dir_y, p_dir_z] = kernels.Scatter(scatter_event_index, scat_count, p_dir_x, p_dir_y, p_dir_z, rands)
    
    end = timer()
    time_scatter = end-start
    start = timer()
    
    fis_count = 0
    [p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, 
         p_time, p_alive, particles_added_fission] = kernels.FissionsAdd(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, 
                                                  p_dir_y, p_dir_z, p_dir_x, p_speed, 
                                                  p_time, p_alive, fis_count, nu_new_neutrons, 
                                                  fission_event_index, num_part, particle_speed, rands)
    
    end = timer()
    time_fission = end-start
    
    [p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, 
         p_time, p_alive, kept] = kernels.BringOutYourDead(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, 
                                                   p_dir_y, p_dir_z, p_dir_x, p_speed, 
                                                   p_time, p_alive, num_part)
    
    end = timer()
    time_BOYD = end-start
    
    end_o = timer()
    time_overall = end_o-start_o
    
    if print_q == True:
        print()
        print('>>>>PRINTING WARMUP TIMES<<<<')
        print('=============================')
        print("Source........{0}".format(time_source))
        print("Advance.......{0}".format(time_ad))
        print("Sample........{0}".format(time_sample))
        print("Still in......{0}".format(time_stillin))
        print("scatter.......{0}".format(time_scatter))
        print("fission.......{0}".format(time_fission))
        print("BOYD..........{0}".format(time_BOYD))
        print()
        print("Overall.......{0}".format(time_overall))
        print()
        
    
if __name__ == '__main__':
    WarmUp()
