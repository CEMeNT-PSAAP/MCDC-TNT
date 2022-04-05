import pyk_kernels.all as kernels
import pykokkos as pk
import numpy as np

def WarmUp():
    pk.set_default_space(pk.ExecutionSpace.OpenMP)
    
    N_mesh = 2
    nu_new_neutrons = 2
    num_part = 1
    dx = 1
    particle_speed = 1
    
    mesh_cap_xsec_np = np.array([1,1], dtype=float)
    mesh_scat_xsec_np =  np.array([1,1], dtype=float)
    mesh_fis_xsec_np =  np.array([1,1], dtype=float)
    mesh_total_xsec_np =  np.array([3,3], dtype=float)
    surface_distances_np =  np.array([0,2], dtype=float)
    
    #===============================================================================
    # Initial setups
    #===============================================================================
    
    # Initialize RNG
    np.random.seed(777)
    
    
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
    
    clever_out: pk.View1D[int] = pk.View([10], pk.int32)
    
    rands_np = np.random.random([num_part*4])
    rands = pk.from_numpy(rands_np)
    
    timer = pk.Timer()
    
    pk.execute(pk.ExecutionSpace.Default, 
        kernels.SourceParticles(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive, num_part, particle_speed, meshwise_fission_pdf, rands))
    
    kernels.Advance(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, num_part, mesh_total_xsec, mesh_dist_traveled, mesh_dist_traveled_squared, surface_distances[len(surface_distances)-1])
    
    pk.execute(pk.ExecutionSpace.Default, kernels.StillIn(p_pos_x, surface_distances, p_alive, num_part, clever_out))
    
    pk.execute(pk.ExecutionSpace.Default, kernels.SampleEvent(p_mesh_cell, p_alive, mesh_cap_xsec, mesh_scat_xsec, mesh_fis_xsec, scatter_event_index,
                                capture_event_index, fission_event_index, num_part, nu_new_neutrons, rands, clever_out))
                                
    scat_count = 0
    pk.execute(pk.ExecutionSpace.Default, kernels.Scatter(scatter_event_index, scat_count, p_dir_x, p_dir_y, p_dir_z, rands))
    
    fis_count = 0
    pk.execute(pk.ExecutionSpace.Default, kernels.FissionsAdd(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, 
                                                  p_dir_y, p_dir_z, p_dir_x, 
                                                  p_time, p_alive, p_speed, fis_count, nu_new_neutrons, 
                                                  fission_event_index, num_part, particle_speed, rands, clever_out))
    
    pk.execute(pk.ExecutionSpace.Default, kernels.BringOutYourDead(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, 
                                                   p_dir_y, p_dir_z, p_dir_x, p_speed, 
                                                   p_time, p_alive, num_part, clever_out))
    
    timer_result = timer.seconds()
    
    return(timer_result)
    
if __name__ == '__main__':
    time = WarmUp()
    
    print(time)
