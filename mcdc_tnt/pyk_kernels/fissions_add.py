"""
Name: FissionsAdd
breif: Adding fission particles to phase vectors for MCDC-TNT
Author: Jackson Morgan (OR State Univ - morgjack@oregonstate.edu) CEMeNT
Date: Nov 18th 2021
"""
import numpy as np
import pykokkos as pk
import math

@pk.workload
class FissionsAdd:
    def __init__(self, p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_time, p_alive, p_speed,
                fis_count, nu_new_neutrons, fission_event_index, num_part, particle_speed, rands):
        self.p_pos_x: pk.View1D[pk.double] = p_pos_x
        self.p_pos_y: pk.View1D[pk.double] = p_pos_y
        self.p_pos_z: pk.View1D[pk.double] = p_pos_z
        
        self.p_dir_x: pk.View1D[pk.double] = p_dir_x
        self.p_dir_y: pk.View1D[pk.double] = p_dir_y
        self.p_dir_z: pk.View1D[pk.double] = p_dir_z
        
        
        self.p_mesh_cell: pk.View1D[int] = p_mesh_cell
        self.p_alive: pk.View1D[int] = p_alive
        self.p_time: pk.View1D[pk.double] = p_time
        self.p_speed: pk.View1D[pk.double] = p_speed
        
        self.fission_event_index: pk.View1D[int] = fission_event_index
        
        self.rands: pk.View1D[pk.double] = rands
        
        self.fis_count: int = fis_count
        self.num_part: int = num_part
        self.nu_new_neutrons: int = nu_new_neutrons
        self.particle_speed: pk.double = particle_speed
        
    
    @pk.main
    def FissionsRun(self):
        k: int = 0 #index for fission temp vectors
        for i in range(self.fis_count):
            for j in range(self.nu_new_neutrons):
                # Position
                self.p_pos_x[k+self.num_part] = self.p_pos_x[self.fission_event_index[i]]
                self.p_pos_y[k+self.num_part] = self.p_pos_y[self.fission_event_index[i]]
                self.p_pos_z[k+self.num_part] = self.p_pos_z[self.fission_event_index[i]]
                
                self.p_mesh_cell[k+self.num_part] = self.p_mesh_cell[self.fission_event_index[i]]
                
                # print("fission particle produced")
                # print("from particle {0} and indexed as particle {1}".format(fission_event_index[i], k+num_part))
                # print("produced at: {0}".format(p_pos_x[k+num_part]))
                # Direction
                # Sample polar and azimuthal angles uniformly
                mu: pk.double  = 2.0*self.rands[4*i+2*j] - 1.0
                azi: pk.double = 2.0*self.rands[4*i+2*j+1]
                # Convert to Cartesian coordinate
                c: pk.double = (1.0 - mu**2)**0.5
                self.p_dir_y[k+self.num_part] = math.cos(azi)*c
                self.p_dir_z[k+self.num_part] = math.sin(azi)*c
                #pk.printf('%f\n',mu)
                self.p_dir_x[k+self.num_part] = mu
                      
                # Speed
                self.p_speed[k+self.num_part] = self.particle_speed
                
                # Time
                self.p_time[k+self.num_part] = self.p_time[self.fission_event_index[i]]

                # Flags
                self.p_alive[k+self.num_part] = 1
                
                k+=1
            
    
    
    

def test_FissionsAdd():
    
    L = 1
    dx = .25
    N_m = 4
    
    num_part = 1
    p_pos_x_np = np.array([.55, 3, 5], dtype=float)
    p_pos_x = pk.from_numpy(p_pos_x_np)
    
    p_pos_y_np = np.array([10, 3, 5], dtype=float)
    p_pos_y = pk.from_numpy(p_pos_y_np)
    
    p_pos_z_np = np.array([15, 3, 5], dtype=float)
    p_pos_z = pk.from_numpy(p_pos_z_np)

    
    
    p_mesh_cell_np = np.array([2, 87, -1], dtype=np.int32)
    
    p_dir_x_np = np.ones(num_part+2, dtype=float)
    p_dir_y_np = np.zeros(num_part+2, dtype=float)
    p_dir_z_np = np.zeros(num_part+2, dtype=float)
    
    p_speed_np = np.ones(num_part+2, dtype=float)
    p_time_np = np.zeros(num_part+2, dtype=float)
    p_alive_np = np.zeros(num_part+2, dtype=np.int32)
    p_alive_np[0] = 1
    
    fis_count = 1
    nu = 2
    particle_speed = 1
    fission_event_index_np = np.array([0,1,413], dtype=np.int32)
    
    rands_np = np.array([1.0,1.0,1.0,1.0,1.0,1.0], dtype=float)
    
    p_dir_x = pk.from_numpy(p_dir_x_np)
    p_dir_y = pk.from_numpy(p_dir_y_np)
    p_dir_z = pk.from_numpy(p_dir_z_np)
    
    p_mesh_cell = pk.from_numpy(p_mesh_cell_np)
    p_speed = pk.from_numpy(p_speed_np)
    p_time = pk.from_numpy(p_time_np)
    p_alive = pk.from_numpy(p_alive_np)
    
    rands = pk.from_numpy(rands_np)
    
    fission_event_index = pk.from_numpy(fission_event_index_np)
    
    
    pk.execute(pk.ExecutionSpace.OpenMP, FissionsAdd(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_time, p_alive, p_speed, fis_count, nu, fission_event_index, num_part, particle_speed, rands))
    
    
    assert(np.allclose(p_pos_x, [0.55, 0.55, 0.55]))
    assert(np.allclose(p_pos_y, [10,10,10]))
    assert(np.allclose(p_pos_z, [15,15,15]))
    assert(p_dir_x[0] == 1)
    assert(np.allclose(p_alive, 1))
    
    
if __name__ == '__main__':
    test_FissionsAdd()
    
