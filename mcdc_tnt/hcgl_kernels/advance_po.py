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
#include <stdio.h>

__global__ void AdvanceOpenCL(float *p_pos_x, float *p_pos_y, float *p_pos_z,
                            float *p_dir_x, float *p_dir_y, float *p_dir_z,
                            int *p_mesh_cell, float *p_speed, float *p_time,
                            float *clever_in, float *mesh_total_xsec,
                            int *p_end_trans, float *rands,
                            float *mesh_dist_traveled, float *mesh_dist_traveled_squared,
                            int *num_dead)
{
    float dx = clever_in[1];
    float L = clever_in[0];
    const int num_part = clever_in[2];
    const int max_mesh_index = clever_in[3];
    
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    //printf("%d\\n", i);
    
    const float kicker = 1e-10;
    const int init_cell = p_mesh_cell[i];
    float p_dist_travled = 0.0;
    int cell_next;
    
    
    
    if (i < num_part){
        
        if (p_end_trans[i] == 0){
            if (p_pos_x[i] < 0){
                p_end_trans[i] = 1;
                atomicAdd(&num_dead[0], 1);
                //printf("%d\\n", i);
                }
                
            else if (p_pos_x[i] >= L){
                p_end_trans[i] = 1;
                atomicAdd(&num_dead[0], 1);
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
                    atomicAdd(&num_dead[0], 1);
                    cell_next = p_mesh_cell[i];
                    //printf("%d\\n", i);
                }
                
                p_pos_x[i] += p_dir_x[i]*p_dist_travled;
                p_pos_y[i] += p_dir_y[i]*p_dist_travled;
                p_pos_z[i] += p_dir_z[i]*p_dist_travled;
                
                atomicAdd(&mesh_dist_traveled[init_cell], p_dist_travled);
                atomicAdd(&mesh_dist_traveled_squared[init_cell], pow(p_dist_travled,2));
                
                p_mesh_cell[i] = cell_next;
                p_time[i]  += p_dist_travled/p_speed[i];
            }
        }
    }
}
""").build
    

def Advance(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time,
            num_part, mesh_total_xsec, mesh_dist_traveled, mesh_dist_traveled_squared, L):
    
    
    
    p_end_trans = np.zeros(num_part, dtype=np.int32)
    end_flag = 0
    max_mesh_index = len(mesh_total_xsec)-1
    
    cycle_count = 0
    
    #copy data to cuda device
    d_p_pos_x = cl_array.to_device(p_pos_x.nbytes)
    d_p_pos_y = cl_array.to_device(p_pos_y.nbytes)
    d_p_pos_z =  cl_array.to_device(p_pos_z.nbytes)
    
    
    d_p_dir_y = cl_array.to_device(p_dir_y.nbytes)
    d_p_dir_z = cl_array.to_device(p_dir_z.nbytes)
    d_p_dir_x = cl_array.to_device(p_dir_x.nbytes)
    
    d_p_mesh_cell = cl_array.to_device(p_mesh_cell.nbytes)
    d_p_speed = cl_array.to_device(p_speed.nbytes)
    d_p_time = cl_array.to_device(p_time.nbytes)
    
    d_p_end_trans = cl_array.to_devicec(p_end_trans.nbytes)
    d_mesh_total_xsec = cl_array.to_device(mesh_total_xsec.nbytes)
    
    d_mesh_dist_traveled =  cl_array.to_device(mesh_dist_traveled.nbytes)
    d_mesh_dist_traveled_squared =  cl_array.to_device(mesh_dist_traveled_squared.nbytes)
    
    
    summer = num_part
    
    number_done = np.zeros(1, dtype=np.int32)
    d_number_done =  cl_array.to_device(number_done.nbytes)
    
    #d_number_done = cuda.to_device(number_done)
    
    AdvanceOpenCL = prg.AdvanceOpenCL
    
    clever_io = np.array([L, dx, num_part, max_mesh_index], np.float32)
    
    while end_flag == 0 and cycle_count < 1000:
        #allocate randoms
        start = timer()
        rands = np.random.random(num_part).astype(np.float32)

        AdvanceOpenCL(queue, p_pos_x.shape, None, d_p_pos_x, d_p_pos_y, d_p_pos_z,
                      d_p_dir_y, d_p_dir_z, d_p_dir_x, 
                      d_p_mesh_cell, d_p_speed, d_p_time,  
                      drv.In(clever_io), d_mesh_total_xsec,
                      d_p_end_trans, drv.In(rands), d_mesh_dist_traveled, d_mesh_dist_traveled_squared, d_number_done)
                      
        drv.memcpy_dtoh(number_done, d_number_done)
        #print(number_done)
        drv.memcpy_dtoh(p_end_trans, d_p_end_trans)
        #print(p_end_trans)
        
        number_done_2 = sum(p_end_trans)
        #print(number_done_2)

        if (number_done[0] == num_part):
            end_flag = 1
        
        end = timer()
        cycle_count += 1
        print("Number done (atomics): {0}    Number done (classical): {1} and took {2}".format(number_done[0], number_done_2, end-start))
        
        #print("Advance Complete:......{0}%       ({1}/{2})    cycle: {3}".format(int(100*number_done[0]/num_part), number_done, num_part, cycle_count), end = "\r")
    print()
        
    
    drv.memcpy_dtoh(p_pos_x, d_p_pos_x)
    drv.memcpy_dtoh(p_pos_y, d_p_pos_y)
    drv.memcpy_dtoh(p_pos_z, d_p_pos_z)
    drv.memcpy_dtoh(p_dir_x, d_p_dir_x)
    drv.memcpy_dtoh(p_dir_y, d_p_dir_y)
    drv.memcpy_dtoh(p_dir_z, d_p_dir_z)
    
    drv.memcpy_dtoh(p_speed, d_p_speed)
    drv.memcpy_dtoh(p_time, d_p_time)
    drv.memcpy_dtoh(p_mesh_cell, d_p_mesh_cell)
    drv.memcpy_dtoh(mesh_dist_traveled, d_mesh_dist_traveled)
    drv.memcpy_dtoh(mesh_dist_traveled_squared, d_mesh_dist_traveled_squared)
    
    
    
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
