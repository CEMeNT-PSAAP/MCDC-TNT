import mcdc_tnt.numba_kernels.cpu as kernels
import numpy as np
import math
  
    
def test_SourceParticles():
    num_parts = 5
    p_pos_x = np.zeros(num_parts)
    p_pos_y = np.zeros(num_parts)
    p_pos_z = np.zeros(num_parts)
    
    p_mesh_cell = np.zeros(num_parts)
    
    p_dir_x = np.zeros(num_parts)
    p_dir_y = np.zeros(num_parts)
    p_dir_z = np.zeros(num_parts)
    
    p_speed = np.zeros(num_parts)
    p_time = np.ones(num_parts)
    p_alive = np.zeros(num_parts)
    
    particle_speed = 1
    meshwise_fission_pdf = [0,1]
    
    iso=False
    
    dx = 0.2
    
    [p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive] = kernels.SourceParticles(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive, num_parts, meshwise_fission_pdf, particle_speed, iso)
    
    assert (np.sum(p_time) == 0)
    assert (p_mesh_cell.all() == 1)
    assert (p_alive.all() == True)
    assert (p_pos_x.all() > .2)





def test_SampleEvent():
    p_mesh_cell = np.array([0,1,0,5])
    p_alive = [True,True,True,False]
    
    mesh_cap_xsec = 1/3*np.ones(2)
    mesh_scat_xsec = 1/3*np.ones(2)
    mesh_fis_xsec = 1/2*np.ones(2)
    
    scatter_event_index = np.zeros(3)
    capture_event_index = np.zeros(3)
    fission_event_index = np.zeros(3) 
    
    controled_rands = np.array([.2, .4, .8])
    
    nu = 2
    num_part = 3
    
    [scatter_event_index, scat_count, capture_event_index, cap_count, fission_event_index, fis_count] = kernels.SampleEvent(p_mesh_cell, p_alive, mesh_cap_xsec, mesh_scat_xsec, mesh_fis_xsec, scatter_event_index, capture_event_index, fission_event_index, num_part, nu, controled_rands)
    
    assert (fis_count == 1)
    assert (scat_count == 1)
    assert (cap_count == 1)
    
    assert (capture_event_index[0] == 1)
    assert (fission_event_index[0] == 2)
    assert (scatter_event_index[0] == 0)
        
        
        
        
        
def test_StillIn():    
    
    num_part = 7
    surface_distances = np.array([0,.25,.75,1], dtype=float)
    p_pos_x = np.array([-.01, 0, .1544, .2257, .75, 1.1, 1], dtype=float)
    p_alive = np.ones(num_part, bool)
    print('Hello!!!!!')
    
    [p_alive, tally_left, tally_right] = kernels.StillIn(p_pos_x, surface_distances, p_alive, num_part)
    
    assert(p_alive[0] == False)
    assert(p_alive[5] == False)
    assert(tally_left == 2)
    assert(tally_right == 2)
    assert(p_alive[2:4].all() == True)
    
    
    
    
    
def test_BOYD():
    
    num_part = 3
    
    p_pos_x = [1,2,3]
    p_pos_y = [1,2,3]
    p_pos_z = [1,2,3]
    
    p_mesh_cell = [1,2,3]
    
    p_dir_x = [1,2,3]
    p_dir_y = [1,2,3]
    p_dir_z = [1,2,3]
    
    p_speed = [1,2,3]
    p_time = [1,2,3]
    p_alive = [False,True,False]
    
    
    [p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive, kept] = kernels.BringOutYourDead(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive, num_part)
    
    assert(kept == 1)
    assert(p_dir_x[0] == 2)
    assert(p_dir_y[0] == 2)
    assert(p_dir_z[0] == 2)
    
    assert(p_pos_x[0] == 2)
    assert(p_pos_y[0] == 2)
    assert(p_pos_z[0] == 2)
    
    assert(p_speed[0] == 2)
    assert(p_time[0] == 2)
    assert(p_alive[0] == True)
    
    
    
    
    
def test_FissionsAdd():
    
    L = 1
    dx = .25
    N_m = 4
    
    num_part = 3
    p_pos_x = np.array([.55, 3, 5])
    p_pos_y = np.array([10, 3, 5])
    p_pos_z = np.array([15, 3, 5])
    
    p_mesh_cell = np.array([2, 87, -1])
    
    p_dir_x = np.ones(num_part)
    p_dir_y = np.zeros(num_part)
    p_dir_z = np.zeros(num_part)
    
    p_speed = np.ones(num_part)
    p_time = np.zeros(num_part)
    p_alive = np.ones(num_part, bool)
    p_alive[0] = False
    
    fis_count = 1
    nu = 2
    fission_event_index = [0]
    
    rands = [1,1,1,1]
    
    [p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive, k] = kernels.FissionsAdd(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive, fis_count, nu, fission_event_index, 1, 1, rands)
    
    print(p_pos_x)
    print(p_pos_y)
    print(p_pos_z)
    
    assert(np.allclose(p_pos_x, [0.55, 0.55, 0.55]))
    assert(np.allclose(p_pos_y, [10,10,10]))
    assert(np.allclose(p_pos_z, [15,15,15]))
    assert(p_dir_x.all() == 1)
    assert(p_alive[1:2].all() == True)
    
    
def test_Advance():
    L = 1
    dx = .25
    N_m = 4
    
    num_part = 6
    p_pos_x = np.array([-.01, 0, .1544, .2257, .75, 1.1])
    p_pos_y = 2.1*np.ones(num_part)
    p_pos_z = 3.4*np.ones(num_part)
    
    p_mesh_cell = np.array([-1, 0, 0, 1, 3, 4])
    
    p_dir_x = np.ones(num_part)
    p_dir_x[0] = -1
    p_dir_y = np.zeros(num_part)
    p_dir_z = np.zeros(num_part)
    
    p_speed = np.ones(num_part)
    p_time = np.zeros(num_part)
    p_alive = np.ones(num_part, bool)
    p_alive[5] = False
    
    
    particle_speed = 1
    mesh_total_xsec = np.array([0.1,1,.1,100])
    
    mesh_dist_traveled_squared = np.zeros(N_m)
    mesh_dist_traveled = np.zeros(N_m)
    
    [p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, mesh_dist_traveled, mesh_dist_traveled_squared] = kernels.Advance(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, num_part, mesh_total_xsec, mesh_dist_traveled, mesh_dist_traveled_squared, L)
    
    
    assert (np.sum(mesh_dist_traveled) > 0)
    assert (np.sum(mesh_dist_traveled_squared) > 0)
    assert (p_pos_x[0]  == -.01)
    assert (p_pos_x[5]  == 1.1)
    assert (p_pos_x[1:4].all()  > .75)
    
    
    
if __name__ == '__main__':
    test_SourceParticles()
    test_SampleEvent()
    test_StillIn()
    test_BOYD()
    test_FissionsAdd()
    test_Advance()
    test_Advance()
    
