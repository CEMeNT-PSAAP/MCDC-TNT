import math
import numpy as np
import pykokkos as pk



@pk.workload
class Advance_cycle:
    def __init__(self, num_part, p_pos_x, p_pos_y, p_pos_z, p_dir_y, p_dir_z, p_dir_x,  p_mesh_cell, p_speed, p_time, dx, mesh_total_xsec, L, p_dist_travled, p_end_trans, rands):
    
        self.p_pos_x: pk.View1D[pk.double] = p_pos_x
        self.p_pos_y: pk.View1D[pk.double] = p_pos_y
        self.p_pos_z: pk.View1D[pk.double] = p_pos_z
        
        self.p_dir_y: pk.View1D[pk.double] = p_dir_y
        self.p_dir_z: pk.View1D[pk.double] = p_dir_z
        self.p_dir_x: pk.View1D[pk.double] = p_dir_x
        
        self.p_mesh_cell: pk.View1D[int] = p_mesh_cell
        self.p_speed: pk.View1D[pk.double] = p_speed
        self.p_time: pk.View1D[pk.double] = p_time
        
        self.dx: pk.double = dx
        self.L: pk.double = L
        #print(dx)
        #print(L)
        self.num_part: int = num_part
        
        self.mesh_total_xsec: pk.View1D[pk.double] = mesh_total_xsec
        
        self.p_dist_travled: pk.View1D[pk.double] = p_dist_travled
        self.p_end_trans: pk.View1D[int] = p_end_trans
        self.rands: pk.View1D[pk.double] = rands
        
    @pk.main
    def run(self):
        pk.parallel_for(self.num_part, self.advanceCycle_wu)
    
    @pk.workunit
    def advanceCycle_wu(self, i: int):
        kicker: pk.double = 1e-8
       
        if (self.p_end_trans[i] == 0):
            if (self.p_pos_x[i] < 0): #exited rhs
                self.p_end_trans[i] = 1
            elif (self.p_pos_x[i] >= self.L): #exited lhs
                self.p_end_trans[i] = 1
                
            else:
                dist: pk.double = -math.log(self.rands[i]) / self.mesh_total_xsec[self.p_mesh_cell[i]]
                
                #pk.printf('%d   %f    %f     %f\n', i, dist, rands[i], mesh_total_xsec[p_mesh_cell[i]])
                
                #p_dist_travled[i] = dist
                
                x_loc: pk.double = (self.p_dir_x[i] * dist) + self.p_pos_x[i]
                LB: pk.double = self.p_mesh_cell[i] * self.dx
                RB: pk.double = LB + self.dx
                
                if (x_loc < LB):        #move partilce into cell at left
                    self.p_dist_travled[i] = (LB - self.p_pos_x[i])/self.p_dir_x[i] + kicker
                    self.p_mesh_cell[i] -= 1
                   
                elif (x_loc > RB):      #move particle into cell at right
                    self.p_dist_travled[i] = (RB - self.p_pos_x[i])/self.p_dir_x[i] + kicker
                    self.p_mesh_cell[i] += 1
                    
                else:                   #move particle in cell
                    self.p_dist_travled[i] = dist
                    self.p_end_trans[i] = 1
                  
                #pk.printf('%d:  x pos before step     %f\n', i, p_pos_x[i])
                self.p_pos_x[i] = self.p_dir_x[i]*self.p_dist_travled[i] + self.p_pos_x[i]
                self.p_pos_y[i] = self.p_dir_y[i]*self.p_dist_travled[i] + self.p_pos_y[i]
                self.p_pos_z[i] = self.p_dir_z[i]*self.p_dist_travled[i] + self.p_pos_z[i]
                
                #pk.printf('%d:  x pos after step:     %f       should be: %f\n', i, p_pos_x[i], (temp_x))
                self.p_time[i]  += dist/self.p_speed[i]


@pk.workload
class DistTraveled:
    def __init__(self, num_part, max_mesh_index, mesh_dist_traveled_pk, mesh_dist_traveled_squared_pk, p_dist_travled, mesh, p_end_trans, clever_out):
        self.num_part: int = num_part
        self.max_mesh_index: int = max_mesh_index
        self.mesh_dist_traveled_pk: pk.View1D[pk.double] = mesh_dist_traveled_pk
        self.mesh_dist_traveled_squared_pk: pk.View1D[pk.double] = mesh_dist_traveled_squared_pk
        self.p_dist_travled: pk.View1D[pk.double] = p_dist_travled
        self.mesh: pk.View1D[int] = mesh
        self.p_end_trans: pk.View1D[int] = p_end_trans
        self.clever_out: pk.View1D[int] = clever_out
    
    @pk.main
    def distTraveled_main(self):
        end_flag: int = 1
        cur_cell: int = 0
        summer: int = 0
        #pk.printf('1   %d\n', cur_cell)
        #pk.printf('3   %f\n', mesh_dist_traveled_pk[cur_cell])
        
        for i in range(self.num_part):
            cur_cell = int(self.mesh[i])
            if (0 < cur_cell) and (cur_cell < self.max_mesh_index):
                self.mesh_dist_traveled_pk[cur_cell] += self.p_dist_travled[i]
                self.mesh_dist_traveled_squared_pk[cur_cell] += self.p_dist_travled[i]**2
                
            if self.p_end_trans[i] == 0:
                end_flag = 0
                
            summer += p_end_trans[i]
            
        clever_out[0] = end_flag
        clever_out[1] = summer

#@pk.workunit
#def CellSum
#    for i in range(num_parts)

    
#@profile
def Advance(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time,
            num_part, mesh_total_xsec, mesh_dist_traveled, mesh_dist_traveled_squared, L):
            
    
    max_mesh_index = int(len(mesh_total_xsec)-1)
    
    p_end_trans: pk.View1D[int] = pk.View([num_part], int) #flag
    p_end_trans.fill(0)
    p_dist_travled: pk.View1D[pk.double] = pk.View([num_part], pk.double)
    
    clever_out: pk.View1D[int] = pk.View([4], int)
    
    end_flag = 0
    cycle_count = 0
    
    while end_flag == 0:
        #allocate randoms
        summer = 0
        rands_np = np.random.random([num_part])
        rands = pk.from_numpy(rands_np)
        #vector of indicies for particle transport
        
        p = pk.RangePolicy(pk.get_default_space(), 0, num_part)
        p_dist_travled.fill(0)
        
        pre_p_mesh = p_mesh_cell
        L = float(L)
        
        #space = pk.ExecutionSpace.OpenMP
        pk.execute(pk.ExecutionSpace.OpenMP, Advance_cycle(num_part, p_pos_x, p_pos_y, p_pos_z, p_dir_y, p_dir_z, p_dir_x, p_mesh_cell, p_speed, p_time, dx, mesh_total_xsec, L, p_dist_travled, p_end_trans, rands))#pk for number still in transport
        
        pk.execute(pk.ExecutionSpace.OpenMP,
            DistTraveled(num_part, max_mesh_index, mesh_dist_traveled, mesh_dist_traveled_squared, p_dist_travled, pre_p_mesh, p_end_trans, clever_out))
        
        end_flag = clever_out[0]
        summer = clever_out[1]
        
        #print(cycle_count)
        if (cycle_count > int(1e3)):
            print("************ERROR**********")
            print(" Max itter hit")
            print(p_end_trans)
            print()
            print()
            return()
        cycle_count += 1
        
        print("Advance Complete:......{1}%       ".format(cycle_count, int(100*summer/num_part)), end = "\r")
    print()
    



@pk.workload
class StillIn:
    def __init__(self, p_pos_x, surface_distances, p_alive, num_part, clever_out):
    
        self.p_pos_x: pk.View1D[pk.double] = p_pos_x
        self.clever_out: pk.View1D[int] = clever_out
        self.surface_distances: pk.View1D[pk.double] = surface_distances
        self.p_alive: pk.View1D[int] = p_alive
        self.num_part: int = num_part
        
    @pk.main
    def run(self):
        tally_left: int = 0
        tally_right: int = 0
        for i in range(self.num_part):
            #exit at left
            if self.p_pos_x[i] <= 0:
                tally_left += 1
                self.p_alive[i] = 0
                
            elif self.p_pos_x[i] >= 1:
                tally_right += 1
                self.p_alive[i] = 0
        
        self.clever_out[0] = tally_left
        self.clever_out[1] = tally_right

def speedTestAdvance():
    # Position
    num_part = int(1e8)
    phase_parts = num_parts
    
    p_pos_x_np = np.zeros(phase_parts, dtype=float)
    p_pos_y_np = np.zeros(phase_parts, dtype=float)
    p_pos_z_np = np.zeros(phase_parts, dtype=float)
    
    p_pos_x = pk.from_numpy(p_pos_x_np)
    p_pos_y = pk.from_numpy(p_pos_y_np)
    p_pos_z = pk.from_numpy(p_pos_z_np)
    
    # Direction
    p_dir_x_np = np.zeros(phase_parts, dtype=float)
    p_dir_y_np = np.zeros(phase_parts, dtype=float)
    p_dir_z_np = np.zeros(phase_parts, dtype=float)
    
    p_dir_x = pk.from_numpy(p_dir_x_np)
    p_dir_y = pk.from_numpy(p_dir_y_np)
    p_dir_z = pk.from_numpy(p_dir_z_np)
    
    # Speed
    p_speed_np = np.zeros(phase_parts, dtype=float)
    p_speed = pk.from_numpy(p_speed_np)
    
    # Time
    p_time_np = np.zeros(phase_parts, dtype=float)
    p_time = pk.from_numpy(p_time_np)
    
    # Region
    p_mesh_cell_np = np.zeros(phase_parts, dtype=np.int32)
    p_mesh_cell = pk.from_numpy(p_mesh_cell_np)
    
    # Flags
    p_alive_np = np.full(phase_parts, False, dtype=np.int32)
    p_alive = pk.from_numpy(p_alive_np)
    
    
    kernels.Advance(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, num_part, mesh_total_xsec, mesh_dist_traveled, mesh_dist_traveled_squared, surface_distances[len(surface_distances)-1])
    
    
"""   
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
    
    [p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, mesh_dist_traveled, mesh_dist_traveled_squared] = Advance(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, num_part, mesh_total_xsec, mesh_dist_traveled, mesh_dist_traveled_squared, L)
    
    
    assert (np.sum(mesh_dist_traveled) > 0)
    assert (np.sum(mesh_dist_traveled_squared) > 0)
    assert (p_pos_x[0]  == -.01)
    assert (p_pos_x[5]  == 1.1)
    assert (p_pos_x[1:4].all()  > .75)
"""
    
        
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
    speedTestAdvance()
