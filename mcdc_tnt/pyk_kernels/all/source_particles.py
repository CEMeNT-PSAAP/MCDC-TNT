"""
Name: CleanUp
breif: Misc functions for MCDC-TNT
Author: Jackson Morgan (OR State Univ - morgjack@oregonstate.edu) CEMeNT
Date: Dec 2nd 2021
"""

import numpy as np
import pykokkos as pk
import math
#import numba as nb

@pk.workload
class SourceParticles:
    def __init__(self, p_pos_x, p_pos_y, p_pos_z,
                p_mesh_cell, dx,
                p_dir_y, p_dir_z, p_dir_x,
                p_speed, p_time, p_alive,
                num_parts, particle_speed, meshwise_fission_pdf, rands):
        
        self.p_pos_x: pk.View1D[pk.double] = p_pos_x
        self.p_pos_y: pk.View1D[pk.double] = p_pos_y
        self.p_pos_z: pk.View1D[pk.double] = p_pos_z
        
        self.p_dir_x: pk.View1D[pk.double] = p_dir_x
        self.p_dir_y: pk.View1D[pk.double] = p_dir_y
        self.p_dir_z: pk.View1D[pk.double] = p_dir_z
        
        self.p_mesh_cell: pk.View1D[int] = p_mesh_cell
        self.p_speed: pk.View1D[pk.double] = p_speed
        self.p_time: pk.View1D[pk.double] = p_time
        self.p_alive: pk.View1D[int] = p_alive
        
        #self.meshwise_fission_pdf: pk.View1D[pk.double] = meshwise_fission_pdf
        
        self.rands: pk.View1D[pk.double] = rands
        self.meshwise_fission_pdf: pk.View1D[pk.double] = meshwise_fission_pdf
        
        self.num_parts: int = num_parts
        self.dx: pk.double = dx
        self.particle_speed: float = particle_speed
        
        #print(meshwise_fission_pdf)
    
    @pk.main
    def run(self):
        #pk.printf("In main\n")
        #pk.printf("\f\n",meshwise_fission_pdf[1])
        #p = pk.RangePolicy(pk.get_default_space(), 0, num_parts)
        #T = pk.Timer()
        pk.parallel_for(self.num_parts, self.sourceP)
        #pr: pk.double = T.seconds()
        
        #pk.printf('%f\n',pr)
        
        
        #pk.printf('%d\n',i)
        #find mesh cell birth based on provided pdf+
    @pk.workunit
    def sourceP(self, i: int):
        cell: int = 0
        summer: float = 0.0
        while (summer < self.rands[i*4]):
            summer += self.meshwise_fission_pdf[cell]
            cell += 1
                
        cell -=1
        
        #pk.printf('%f\n',self.rands[i*4])
        self.p_mesh_cell[i] = int(cell)
        
        #sample birth location within cell
        self.p_pos_x[i] = dx*cell + dx*self.rands[i*4+1]
        self.p_pos_y[i] = 0.0
        self.p_pos_z[i] = 0.0
        
        
        # Sample polar and azimuthal angles uniformly
        mu: pk.double  = 2.0*self.rands[i*4+2] - 1.0
        azi: pk.double = 2.0*self.rands[i*4+3]
    
        # Convert to Cartesian coordinate
        c: pk.double = (1.0 - mu**2)**0.5
        self.p_dir_y[i] = math.cos(azi)*c
        self.p_dir_z[i] = math.sin(azi)*c
        self.p_dir_x[i] = mu

        # Speed
        self.p_speed[i] = particle_speed

        # Time
        self.p_time[i] = 0.0
        
        self.p_alive[i] = 1
    
    #@pk.callback
    #def ReturnSource(self):
        #print("ALL done!")
        #return(self.p_pos_x, self.p_pos_y, self.p_pos_z, self.p_mesh_cell, self.p_dir_y, self.p_dir_z, self.p_dir_x, self.p_speed, self.p_time)



def test_SourceParticles():
    num_parts = 5
    p_pos_x_np = np.zeros(num_parts)
    p_pos_y_np = np.zeros(num_parts)
    p_pos_z_np = np.zeros(num_parts)
    
    p_mesh_cell_np = np.zeros(num_parts, dtype=np.int32)
    
    p_dir_x_np = np.zeros(num_parts)
    p_dir_y_np = np.zeros(num_parts)
    p_dir_z_np = np.zeros(num_parts)
    
    p_speed_np = np.zeros(num_parts)
    p_time_np = np.ones(num_parts)
    p_alive_np = np.zeros(num_parts, dtype=np.int32)
    
    particle_speed = 1
    meshwise_fission_pdf_np = np.array([0.0,1.001], dtype=float)
    
    iso=False
    
    dx = 0.2
    
    p_pos_x = pk.from_numpy(p_pos_x_np)
    p_pos_y = pk.from_numpy(p_pos_y_np)
    p_pos_z = pk.from_numpy(p_pos_z_np)
    
    p_dir_x = pk.from_numpy(p_dir_x_np)
    p_dir_y = pk.from_numpy(p_dir_y_np)
    p_dir_z = pk.from_numpy(p_dir_z_np)
    
    p_mesh_cell = pk.from_numpy(p_mesh_cell_np)
    p_speed = pk.from_numpy(p_speed_np)
    p_time = pk.from_numpy(p_time_np)
    p_alive = pk.from_numpy(p_alive_np)
    
    meshwise_fission_pdf = pk.from_numpy(meshwise_fission_pdf_np)
    
    rands_np = np.random.random(4*num_parts)
    rands = pk.from_numpy(rands_np)
    
    #p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive] = 
    pk.execute(pk.ExecutionSpace.OpenMP, Source_Particles(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive, num_parts, particle_speed,  meshwise_fission_pdf, rands))
    
    #[p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive] = SourceParticles(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive, num_parts, meshwise_fission_pdf, particle_speed, rands)
    
    print("Ran")
    
    assert (np.sum(p_time) == 0)
    assert (np.allclose(p_mesh_cell, 1))
    assert (np.allclose(p_alive, 1))
    assert (p_pos_x[3] > .2)
    
if __name__ == '__main__':
    test_SourceParticles()    

