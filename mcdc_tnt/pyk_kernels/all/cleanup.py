"""
Name: CleanUp
breif: Misc functions for MCDC-TNT
Author: Jackson Morgan (OR State Univ - morgjack@oregonstate.edu) CEMeNT
Date: Dec 2nd 2021
"""
import pykokkos as pk
import numpy as np

@pk.workload
class BringOutYourDead:
    def __init__ (self,p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive, num_part, clever_out):
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
        
        self.num_part: int = num_part
        
        self.clever_out: pk.View1D[int] = clever_out
        
    @pk.main
    def BOYD(self):
        kept: int = 0
        for i in range(num_part):
            if p_alive[i] == 1:
                self.p_pos_x[kept] = self.p_pos_x[i]
                self.p_pos_y[kept] = self.p_pos_y[i]
                self.p_pos_z[kept] = self.p_pos_z[i]
                
                # Direction
                self.p_dir_x[kept] = self.p_dir_x[i]
                self.p_dir_y[kept] = self.p_dir_y[i]
                self.p_dir_z[kept] = self.p_dir_z[i]
                
                # Speed
                self.p_speed[kept] = self.p_speed[i]
                
                # Time
                self.p_time[kept] = self.p_time[i]
                
                # Regions
                self.p_mesh_cell[kept] = self.p_mesh_cell[i]
                
                # Flags
                self.p_alive[kept] = self.p_alive[i] 
                kept +=1
        #pk.printf('>>>>> %d\n particles kept', kept)
        self.clever_out[0] = kept
                
    #@pk.callback
    #def ReturnKept(self):
    #    self.clever_out = self.kept
    
    
def test_BOYD():
    
    num_part = 3
    
    p_pos_x_np = np.array([1,2,3], dtype=float)
    p_pos_y_np = np.array([1,2,3], dtype=float)
    p_pos_z_np = np.array([1,2,3], dtype=float)
    
    p_mesh_cell_np = np.array([1,2,3], dtype=np.int32)
    
    p_dir_x_np = np.array([1,2,3], dtype=float)
    p_dir_y_np = np.array([1,2,3], dtype=float)
    p_dir_z_np = np.array([1,2,3], dtype=float)
    
    p_speed_np = np.array([1,2,3], dtype=float)
    p_time_np = np.array([1,2,3], dtype=float)
    p_alive_np = np.array([0,1,0], dtype=np.int32)
    
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
    
    clever_out_np = np.array([0], dtype=np.int32)
    clever_out = pk.from_numpy(clever_out_np)
    
    pk.execute(pk.ExecutionSpace.OpenMP,
        BringOutYourDead(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_alive, num_part, clever_out))
    
    kept = clever_out[0]
    
    print(kept)
    
    print(p_pos_x)
    
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
    
if __name__ == '__main__':
    test_BOYD()

