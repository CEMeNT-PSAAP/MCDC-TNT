"""
Name: Scatter
breif: Adding fission particles to phase vectors for MCDC-TNT
Author: Jackson Morgan (OR State Univ - morgjack@oregonstate.edu) CEMeNT
Date: Dec 2nd 2021
"""

import math
import numpy as np
import pykokkos as pk

@pk.workload
class Scatter:
    def __init__(self,scatter_indices, scat_count, p_dir_x, p_dir_y, p_dir_z, rands):
        self.scatter_indices: pk.View1D[int] = scatter_indices
        self.scat_count: int = scat_count
        self.p_dir_x: pk.View1D[pk.double] = p_dir_x
        self.p_dir_y: pk.View1D[pk.double] = p_dir_y
        self.p_dir_z: pk.View1D[pk.double] = p_dir_z
        self.rands: pk.View1D[float] = rands 
    
    
    @pk.main
    def run(self):
        pk.parallel_for(self.scat_count, self.Scatter_wu)
    
    @pk.callback
    def ReturnScatter(self):
        print("All done!")
        
    @pk.workunit
    def Scatter_wu(self, i: int):
        # Sample polar and azimuthal angles uniformly
        mu: pk.double  = 2.0*self.rands[2*i] - 1.0
        azi: pk.double = 2.0*3.14159265359*self.rands[2*i+1]
	    
        # Convert to Cartesian coordinate
        c: pk.double = (1.0 - mu**2)**0.5
        self.p_dir_y[self.scatter_indices[i]] = math.cos(azi)*c
        self.p_dir_z[self.scatter_indices[i]] = math.sin(azi)*c
        self.p_dir_x[self.scatter_indices[i]] = mu
    
    
    
def test_Scatter():
    
    scat_count = 3
    scatter_indices = np.array([0,1,4], dtype=np.int32)
    p_dir_x = np.array([1,2,0,0,4], dtype=float)
    p_dir_y = np.array([1,2,0,0,4], dtype=float)
    p_dir_z = np.array([1,2,0,0,4], dtype=float)
    rands = np.array([1,1,0,0,.5,.5], dtype=float)
    
    p_dir_x = pk.from_numpy(p_dir_x)
    p_dir_y = pk.from_numpy(p_dir_y)
    p_dir_z = pk.from_numpy(p_dir_z)
    
    rands = pk.from_numpy(rands)
    
    scatter_indices = pk.from_numpy(scatter_indices)
    
    
    
    pk.execute(pk.ExecutionSpace.OpenMP, Scatter(scatter_indices, scat_count, p_dir_x, p_dir_y, p_dir_z, rands))
    
    
    assert(p_dir_y[0] == 0)
    assert(p_dir_z[0] == 0)
    assert(p_dir_x[0] == 1)
    
    assert(p_dir_y[1] == 0)
    assert(p_dir_z[1] == 0)
    assert(p_dir_x[1] == -1)
    
    assert(p_dir_y[4] == -1)
    assert(np.allclose(p_dir_z[4], 0))
    assert(p_dir_x[4] == 0)
    
    print("Passed!")
    
if __name__ == '__main__':
    test_Scatter()
