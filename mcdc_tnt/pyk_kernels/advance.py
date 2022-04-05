import math
import numpy as np
import pykokkos as pk


def Randoms(num_parts):
    return(np.random.random(num_parts))

@pk.workload
class Advance:
    def __init__(self, p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time,
            num_part, mesh_total_xsec, mesh_dist_traveled, mesh_dist_traveled_squared, L):
        self.p_pos_x = pk.from_numpy(p_pos_x)
        self.p_pos_y = pk.from_numpy(p_pos_y)
        self.p_pos_z = pk.from_numpy(p_pos_z)
        
        self.p_dir_x = pk.from_numpy(p_dir_x)
        self.p_dir_y = pk.from_numpy(p_dir_y)
        self.p_dir_z = pk.from_numpy(p_dir_z)
        
        self.p_mesh_cell = pk.from_numpy(p_mesh_cell)
        self.p_speed = pk.from_numpy(p_speed)
        self.p_time = pk.from_numpy(p_time)
        
        self.num_part: int = num_part
        self.dx: pk.double = dx
        self.L: pk.double = L
        
        kicker: pk.double = 1e-8
        
        p_end_trans: pk.View1D[int] = pk.View([num_part], int) #flag
        p_end_trans.fill(0)
        end_flag = 0

        cycle_count = 0
        p_dist_travled: pk.View1D[pk.float] = pk.View([num_part], pk.double)
    
    
    
    @pk.callback
    def return_ad(self):
        return(self.p_pos_x, self.p_pos_y, self.p_pos_z, self.p_mesh_cell, self.p_dir_y, self.p_dir_z, self.p_dir_x, self.p_speed, self.p_time, self.mesh_dist_traveled, self.mesh_dist_traveled_squared)
        
        
    
    @pk.main
    def run(self):
        while end_flag == 0:
            #allocate randoms
            summer = 0
            rands_np = Randoms(num_part)
            rands = pk.from_numpy(rands_np)
            #vector of indicies for particle transport
            
            #p = pk.RangePolicy(pk.get_default_space(), 0, num_part)
            p_dist_travled.fill(0)
            
            pre_p_mesh = p_mesh_cell_pk
            
            pk.parallel_for(self.num_part, self.Advance_cycle, p_dist_traveled=p_dist_traveled, p_end_trans=p_end_trans, rands=rands)#pk for number still in transport
            post_p_x = p_pos_x_pk
            
            end_flag = 1
            for i in range(num_part):
                if (0 < pre_p_mesh[i] < max_mesh_index):
                    mesh_dist_traveled[pre_p_mesh[i]] += p_dist_travled[i]
                    mesh_dist_traveled_squared[pre_p_mesh[i]] += p_dist_travled[i]**2
                    
                if p_end_trans[i] == 0:
                    end_flag = 0
                    
                summer += p_end_trans[i]
            
            #print(cycle_count)
            #if (cycle_count > int(1e3)):
            #    print("************ERROR**********")
            #    print(" Max itter hit")
            #    print(p_end_trans)
            #    print()
            #    print()
            #    return()
            cycle_count += 1
            
            #print("Advance Complete:......{1}%       ".format(cycle_count, int(100*summer/num_part)), end = "\r")
        #print()
        
        
        
    
    @pk.workunit
    def Advance_cycle(i: int, p_dist_travled: pk.View1D[pk.double], p_end_trans: pk.View1D[int], rands: pk.View1D[pk.double]):
        #pk.printf('%d\n', i)
        #pk.printf('%d   %d     %f\n',i,p_mesh_cell[i], p_pos_x[i])
        
        
       
        if (p_end_trans[i] == 0):
            if (p_pos_x[i] < 0): #exited rhs
                p_end_trans[i] = 1
            elif (p_pos_x[i] >= L): #exited lhs
                p_end_trans[i] = 1
                
            else:
                dist: pk.double = -math.log(rands[i]) / self.mesh_total_xsec[self.p_mesh_cell[i]]
                
                #pk.printf('%d   %f    %f     %f\n', i, dist, rands[i], mesh_total_xsec[p_mesh_cell[i]])
                
                #p_dist_travled[i] = dist
                
                x_loc: pk.double = (self.p_dir_x[i] * dist) + self.p_pos_x[i]
                LB: pk.double = self.p_mesh_cell[i] * self.dx
                RB: pk.double = self.LB + self.dx
                
                if (x_loc < LB):        #move partilce into cell at left
                    p_dist_travled[i] = (LB - self.p_pos_x[i])/self.p_dir_x[i] + self.kicker
                    self.p_mesh_cell[i] -= 1
                   
                elif (x_loc > RB):      #move particle into cell at right
                    p_dist_travled[i] = (RB - self.p_pos_x[i])/self.p_dir_x[i] + self.kicker
                    self.p_mesh_cell[i] += 1
                    
                else:                   #move particle in cell
                    p_dist_travled[i] = dist
                    p_end_trans[i] = 1
                  
                #pk.printf('%d:  x pos before step     %f\n', i, p_pos_x[i])
                #p_pos_x[i] = p_dir_x[i]*p_dist_travled[i] + p_pos_x[i]
                self.p_pos_x[i] = self.p_dir_x[i]*p_dist_travled[i] + self.p_pos_x[i]
                self.p_pos_y[i] = self.p_dir_y[i]*p_dist_travled[i] + self.p_pos_y[i]
                self.p_pos_z[i] = self.p_dir_z[i]*p_dist_travled[i] + self.p_pos_z[i]
                
                #pk.printf('%d:  x pos after step:     %f       should be: %f\n', i, p_pos_x[i], (temp_x))
                self.p_time[i]  += dist/self.p_speed[i]



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
    L = 1
    dx = .25
    N_m = 4
    
    num_part = 6
    p_pos_x = np.array([-.01, 0, .1544, .2257, .75, 1.1])
    p_pos_y = 2.1*np.ones(num_part)
    p_pos_z = 3.4*np.ones(num_part)
    
    p_mesh_cell = np.array([-1, 0, 0, 1, 3, 4], dtype=int)
    
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
    
    [p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, mesh_dist_traveled, mesh_dist_traveled_squared] = pk.execute(pk.ExecutionSpace.OpenMP, Advance(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, num_part, mesh_total_xsec, mesh_dist_traveled, mesh_dist_traveled_squared, L))
    
    
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
    test_StillIn()
   

    
