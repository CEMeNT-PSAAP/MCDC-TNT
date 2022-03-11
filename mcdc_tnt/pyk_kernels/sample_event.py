"""
Name: SampleEvent
breif: Samples events for particles provided in a phase space for MCDC-TNT
Author: Jackson Morgan (OR State Univ - morgjack@oregonstate.edu) CEMeNT
Date: Dec 2nd 2021
"""
import numpy as np
import pykokkos as pk

@pk.workload
class SampleEvent:
    def __init__(self, p_mesh_cell, p_alive, mesh_cap_xsec, mesh_scat_xsec, mesh_fis_xsec, scatter_event_index, capture_event_index, fission_event_index, num_part, nu_new_neutrons, rands, clever_out):
        self.p_mesh_cell: pk.View1D[int] = p_mesh_cell
        self.p_alive: pk.View1D[int] = p_alive
        
        self.mesh_cap_xsec: pk.View1D[pk.double] = mesh_cap_xsec
        self.mesh_scat_xsec: pk.View1D[pk.double] = mesh_scat_xsec
        self.mesh_fis_xsec: pk.View1D[pk.double] = mesh_fis_xsec
        
        self.scatter_event_index: pk.View1D[int] = scatter_event_index
        self.capture_event_index: pk.View1D[int] = capture_event_index
        self.fission_event_index: pk.View1D[int] = fission_event_index
        
        self.num_part: int = num_part
        self.nu_new_neutrons: int = num_part
        self.rands: pk.View1D[pk.double] = rands
        
        self.fissions_to_add: int = 0
        self.scat_count: int = 0
        self.cap_count: int = 0
        self.fis_count: int = 0
        self.killed: int = 0
        
        
        
        self.clever_out: pk.View1D[int] = clever_out
        
        #print('made it through init!')

    @pk.main
    def run(self):
        for i in range(self.num_part):
            #normalize cross sections in each mesh cell
            total_scat_xsec: pk.double = self.mesh_scat_xsec[self.p_mesh_cell[i]] + self.mesh_cap_xsec[self.p_mesh_cell[i]] + self.mesh_fis_xsec[self.p_mesh_cell[i]]
            mesh_scat_xsec_temp: pk.double = self.mesh_scat_xsec[self.p_mesh_cell[i]] / total_scat_xsec
            mesh_cap_xsec_temp: pk.double = self.mesh_cap_xsec[self.p_mesh_cell[i]] / total_scat_xsec
            mesh_fis_xsec_temp: pk.double = self.mesh_fis_xsec[self.p_mesh_cell[i]] / total_scat_xsec
            
            #pk.printf('%d     %d     %d\n ',self.scat_count, self.cap_count, self.fis_count)
            
            if self.p_alive[i] == 1:
                
                event_rand:pk.double = self.rands[i]
                
                #scatter?
                if event_rand < mesh_scat_xsec_temp:
                    self.scatter_event_index[self.scat_count] = i
                    self.scat_count += 1
                    #pk.printf('had a scatter! %d\n', self.scat_count)
                
                #capture?
                elif mesh_scat_xsec_temp < event_rand  and event_rand < mesh_scat_xsec_temp + mesh_cap_xsec_temp:
                    self.p_alive[i] = 0
                    self.killed += 1
                    self.capture_event_index[self.cap_count] = i
                    self.cap_count +=1
                    #pk.printf('had a capture! %d\n', self.cap_count)
                    
                #fission?
                elif mesh_scat_xsec_temp + mesh_cap_xsec_temp < event_rand and event_rand < mesh_scat_xsec_temp + mesh_cap_xsec_temp + mesh_fis_xsec_temp:
                    self.p_alive[i] = 0
                    self.killed += 1
                    self.fissions_to_add += self.nu_new_neutrons
                    self.fission_event_index[self.fis_count] = i
                    self.fis_count += 1
                    #pk.printf('had a fission! %d\n', self.fis_count)
                    
                else:
                    pk.printf('Well shoot dang')
                    
        self.clever_out[0] = self.scat_count
        self.clever_out[1] = self.cap_count
        self.clever_out[2] = self.fis_count
    
    
    
def test_SampleEvent():
        p_mesh_cell = np.array([0,1,0,5], dtype=np.int32)
        p_alive = np.array([1,1,1,0], dtype=np.int32)
        
        mesh_cap_xsec = 1/3*np.ones(2, dtype=float)
        mesh_scat_xsec = 1/3*np.ones(2, dtype=float)
        mesh_fis_xsec = 1/2*np.ones(2, dtype=float)
        
        scatter_event_index = np.zeros(3, dtype=np.int32)
        capture_event_index = np.zeros(3, dtype=np.int32)
        fission_event_index = np.zeros(3, dtype=np.int32) 
        
        clever_out = np.zeros(3, dtype=np.int32) 
        
        controled_rands = np.array([.2, .4, .8], dtype=float)
        
        nu = 2
        num_part = 3
        
        p_mesh_cell = pk.from_numpy(p_mesh_cell)
        p_alive = pk.from_numpy(p_alive)
        
        mesh_cap_xsec = pk.from_numpy(mesh_cap_xsec)
        mesh_scat_xsec = pk.from_numpy(mesh_scat_xsec)
        mesh_fis_xsec = pk.from_numpy(mesh_fis_xsec)
        
        scatter_event_index = pk.from_numpy(scatter_event_index)
        capture_event_index = pk.from_numpy(capture_event_index)
        fission_event_index = pk.from_numpy(fission_event_index)
        
        controled_rands = pk.from_numpy(controled_rands)
        
        clever_out = pk.from_numpy(clever_out)
        
        print("Running!")
        pk.execute(pk.ExecutionSpace.OpenMP, SampleEvent(p_mesh_cell, p_alive, mesh_cap_xsec, mesh_scat_xsec, mesh_fis_xsec, scatter_event_index, capture_event_index, fission_event_index, num_part, nu, controled_rands, clever_out))
        print('Made it through')
        
        scat_count = clever_out[0]
        cap_count = clever_out[1]
        fis_count = clever_out[2] 
        
        print(scat_count)
        
        assert (fis_count == 1)
        assert (scat_count == 1)
        assert (cap_count == 1)
        
        
        assert (capture_event_index[0] == 1)
        assert (fission_event_index[0] == 2)
        assert (scatter_event_index[0] == 0)
        
if __name__ == '__main__':
    test_SampleEvent()
