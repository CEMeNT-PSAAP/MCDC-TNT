"""
Name: Advance
breif: inputdeck for MCDC-TNT
Author: Jackson Morgan (OR State Univ - morgjack@oregonstate.edu) CEMeNT
Date: Dec 2nd 2021
"""

import math
import numpy as np
from timeit import default_timer as timer
import pyopencl as cl
import pyopencl.array as cl_array

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

prg = cl.Program(ctx, """
__kernel void AdvanceOpenCl(__global float *p_pos_x, __global float *p_pos_y, __global float *p_pos_z,
                            __global float *p_dir_x, __global float *p_dir_y, __global float *p_dir_z,
                            __global int *p_mesh_cell, __global float *p_speed, __global float *p_time,
                            __global float *clever_in, __global float *mesh_total_xsec,
                            __global int *p_end_trans, __global float *rands,
                            __global float *mesh_dist_traveled, __global float *mesh_dist_traveled_squared,
                            __global int *num_dead)
{
    float dx = clever_in[1];
    float L = clever_in[0];
    const int num_part = clever_in[2];
    const int max_mesh_index = clever_in[3];
    
    int i = get_global_id(0);
    
    const float kicker = 1e-10;
    const int init_cell = p_mesh_cell[i];
    float p_dist_travled = 0.0;
    int cell_next;
    
    
    
    if (i < num_part){
        
        if (p_end_trans[i] == 0){
            if (p_pos_x[i] < 0){
                p_end_trans[i] = 1;
                //atomicAdd(&num_dead[0], 1);
                //printf("%d\\n", i);
                }
                
            else if (p_pos_x[i] >= L){
                p_end_trans[i] = 1;
                //atomicAdd(&num_dead[0], 1);
                //printf("%d\\n", i);
            }
            else{
                float dist = -log(rands[i]/mesh_total_xsec[p_mesh_cell[i]]);
                
                float x_loc = (p_dir_x[i] * dist) + p_pos_x[i];
                float LB = p_mesh_cell[i] * dx;
                float RB = LB + dx;
                
                
                
                if (x_loc < LB){
                    p_dist_travled = (LB - p_pos_x[i])/p_dir_x[i] + kicker; //29
                    cell_next = p_mesh_cell[i] - 1;
                }
                   
                else if (x_loc > RB){
                    p_dist_travled = (RB - p_pos_x[i])/p_dir_x[i] + kicker;
                    cell_next = p_mesh_cell[i] + 1;
                }
                    
                else{
                    p_dist_travled = dist;
                    p_end_trans[i] = 1;
                    //atomicAdd(&num_dead[0], 1);
                    cell_next = p_mesh_cell[i];
                    //printf("%d\\n", i);
                }
                
                p_pos_x[i] += p_dir_x[i]*p_dist_travled;
                p_pos_y[i] += p_dir_y[i]*p_dist_travled;
                p_pos_z[i] += p_dir_z[i]*p_dist_travled;
                
                //atomicAdd(&mesh_dist_traveled[init_cell], p_dist_travled);
                //atomicAdd(&mesh_dist_traveled_squared[init_cell], pow(p_dist_travled,2));
                
                p_mesh_cell[i] = cell_next;
                p_time[i]  += p_dist_travled/p_speed[i];
            }
        }
    }
}

__kernel void TallyMesh(
{
    int i = get_global_id(0);
    
}
""").build()
    

def Advance(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time,
            num_part, mesh_total_xsec, mesh_dist_traveled, mesh_dist_traveled_squared, L):
    
    
    
    p_end_trans = np.zeros(num_part, dtype=np.int32)
    end_flag = 0
    max_mesh_index = len(mesh_total_xsec)-1
    
    cycle_count = 0
    
    #copy data to cuda device
    d_p_pos_x = cl_array.to_device(queue, p_pos_x)
    d_p_pos_y = cl_array.to_device(queue, p_pos_y)
    d_p_pos_z =  cl_array.to_device(queue, p_pos_z)
    
    
    d_p_dir_y = cl_array.to_device(queue, p_dir_y)
    d_p_dir_z = cl_array.to_device(queue, p_dir_z)
    d_p_dir_x = cl_array.to_device(queue, p_dir_x)
    
    d_p_mesh_cell = cl_array.to_device(queue, p_mesh_cell)
    d_p_speed = cl_array.to_device(queue, p_speed)
    d_p_time = cl_array.to_device(queue, p_time)
    
    d_p_end_trans = cl_array.to_device(queue, p_end_trans)
    d_mesh_total_xsec = cl_array.to_device(queue, mesh_total_xsec)
    
    d_mesh_dist_traveled =  cl_array.to_device(queue, mesh_dist_traveled)
    d_mesh_dist_traveled_squared =  cl_array.to_device(queue, mesh_dist_traveled_squared)
    
    
    summer = num_part
    
    number_done = np.zeros(1, dtype=np.int32)
    d_number_done =  cl_array.to_device(queue, number_done)
    
    #d_number_done = cuda.to_device(number_done)
    
    knl = prg.AdvanceOpenCl
    
    clever_io = np.array([L, dx, num_part, max_mesh_index], np.float32)
    d_clever_io = cl_array.to_device(queue, clever_io)
    
    while end_flag == 0 and cycle_count < 1000:
        #allocate randoms
        start = timer()
        rands = np.random.random(num_part).astype(np.float32)
        d_rands = cl_array.to_device(queue, rands)
        
        knl(queue, p_pos_x.shape, None, d_p_pos_x.data, d_p_pos_y.data, d_p_pos_z.data,
                      d_p_dir_y.data, d_p_dir_z.data, d_p_dir_x.data, 
                      d_p_mesh_cell.data, d_p_speed.data, d_p_time.data,  
                      d_clever_io.data, d_mesh_total_xsec.data,
                      d_p_end_trans.data, d_rands.data, d_mesh_dist_traveled.data, d_mesh_dist_traveled_squared.data, d_number_done.data)
                      
        number_done = d_number_done.get()
        #print(number_done)
        p_end_trans = d_p_end_trans.get()
        #print(p_end_trans)
        
        number_done_2 = sum(p_end_trans)
        #print(number_done_2)

        if (number_done_2 == num_part):
            end_flag = 1
        
        end = timer()
        cycle_count += 1
        #print("Number done (atomics): {0}    Number done (classical): {1} and took {2}".format(number_done[0], number_done_2, end-start))
        
        #print("Advance Complete:......{0}%       ({1}/{2})    cycle: {3}".format(int(100*number_done[0]/num_part), number_done, num_part, cycle_count), end = "\r")
    print()
        
    
    p_pos_x = d_p_pos_x.get()
    p_pos_y = d_p_pos_y.get()
    p_pos_z = d_p_pos_z.get()
    p_dir_x = d_p_dir_x.get()
    p_dir_y = d_p_dir_y.get()
    p_dir_z = d_p_dir_z.get()
    
    p_speed = d_p_speed.get()
    p_time = d_p_time.get()
    p_mesh_cell = d_p_mesh_cell.get()
    mesh_dist_traveled = d_mesh_dist_traveled.get()
    mesh_dist_traveled_squared = d_mesh_dist_traveled_squared.get()
    
    
    
    return(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, mesh_dist_traveled, mesh_dist_traveled_squared)

     
                
def StillIn(p_pos_x, surface_distances, p_alive, num_part):
    tally_left = 0
    tally_right = 0
    for i in range(num_part):
        #exit at left
        if p_pos_x[i] <= surface_distances[0]:
            tally_left += 1
            p_alive[i] = False
            
        elif p_pos_x[i] >= surface_distances[len(surface_distances)-1]:
            tally_right += 1
            p_alive[i] = False
            
    return(p_alive, tally_left, tally_right)




def test_Advance():
    L: float = 1
    dx: float = .25
    N_m: int = 4
    
    num_part: int = 6
    p_pos_x = np.array([-.01, 0, .1544, .2257, .75, 1.1], np.float32)
    p_pos_y = 2.1*np.ones(num_part, np.float32)
    p_pos_z = 3.4*np.ones(num_part, np.float32)
    
    p_mesh_cell = np.array([-1, 0, 0, 1, 3, 4], np.int32)
    
    p_dir_x = np.ones(num_part, np.float32)
    p_dir_x[0] = -1
    p_dir_y = np.zeros(num_part, np.float32)
    p_dir_z = np.zeros(num_part, np.float32)
    
    p_speed = np.ones(num_part, np.float32)
    p_time = np.zeros(num_part, np.float32)
    p_alive = np.ones(num_part, np.int32)
    p_alive[5] = 0
    
    
    particle_speed = 1
    mesh_total_xsec = np.array([0.1,1,.1,100], np.float32)
    
    mesh_dist_traveled_squared = np.zeros(N_m, np.float32)
    mesh_dist_traveled = np.zeros(N_m, np.float32)
    
    
    [p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, mesh_dist_traveled, mesh_dist_traveled_squared] = Advance(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, num_part, mesh_total_xsec, mesh_dist_traveled, mesh_dist_traveled_squared, L)
    
    
    assert (np.sum(mesh_dist_traveled) > 0)
    assert (np.sum(mesh_dist_traveled_squared) > 0)
    assert (p_pos_x[0]  == -.01)
    assert (p_pos_x[5]  == 1.1)
    assert (p_pos_x[1:4].all()  > .75)
    
    
        
def test_StillIn():    
    
    num_part = 7
    surface_distances = [0,.25,.75,1]
    p_pos_x = np.array([-.01, 0, .1544, .2257, .75, 1.1, 1])
    p_alive = np.ones(num_part, bool)
    
    [p_alive, tally_left, tally_right] = StillIn(p_pos_x, surface_distances, p_alive, num_part)
    
    assert(p_alive[0] == False)
    assert(p_alive[5] == False)
    assert(tally_left == 2)
    assert(tally_right == 2)
    assert(p_alive[2:4].all() == True)


if __name__ == '__main__':
    test_Advance()
    #test_StillIn()
