import pp_kernels as kernels

class test_pp_kernels:
    #def Advance(self):
        
        
    #    kernels.advance
        
        
    #def Cleanup(self):
    
    
    #def FissionsAdd(self)
    
    
    
    def test_SampleEvent(self):
        p_mesh_cell = np.array[0,1,0,5]
        p_alive = [True,True,True,False]
        
        mesh_cap_xsec = 1/3*np.ones(2)
        mesh_scat_xsec = 1/3*np.ones(2)
        mesh_fis_xsec = 1/2*np.ones(2)
        
        scatter_event_index = np.zeros(3)
        capture_event_index = np.zeros(3)
        fission_event_index = np.zeros(3) 
        
        rands = np.array([.2, .4, .8])
        
        nu = 2
        num_part = 3
        
        [scatter_event_index, scat_count, capture_event_index, cap_count, fission_event_index, fis_count] = SampleEvent(p_mesh_cell, p_event, p_alive, mesh_cap_xsec, mesh_scat_xsec, mesh_fis_xsec, scatter_event_index, capture_event_index, fission_event_index, num_part, no, rands)
        
        assert (fis_count == 1)
        assert (scat_count == 1)
        assert (cap_count == 1)
        
        assert (capture_event_index[0] == 1)
        assert (fission_event_index[1] == 1)
        assert (scatter_event_index[2] == 1)
        
    
    #def Scatter(self):
    
    
    
    #def SourceParticles(self):
    
    
    
