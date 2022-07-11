import math
import numpy as np
import pykokkos as pk


@pk.workload
class Advance_cycle:
    def __init__(self, num_part, p_pos_x, p_pos_y, p_pos_z, p_dir_y, p_dir_z, p_dir_x, p_mesh_cell, p_speed, p_time, p_time_cell, dx, dt, n_mesh, mesh_total_xsec, L, p_end_trans, rands, mesh_dist_traveled, mesh_dist_traveled_squared, max_x, max_time, clever_out):
        #print(p_pos_x.dtype)
        #print(p_pos_y.dtype)
        #print(p_pos_z.dtype)
        #print(p_dir_x.dtype)
        #print(p_dir_y.dtype)
        #print(p_dir_z.dtype)
        #print(p_mesh_cell.dtype)
        #print(p_speed.dtype)
        #print(p_time.dtype)
        #print(rands.dtype)
        #print(p_time_cell.dtype)
        #print(rands.dtype)
        #print(mesh_total_xsec.dtype)
        #print(p_end_trans.dtype)
        #print(mesh_dist_traveled.dtype)
        #print(mesh_dist_traveled_squared.dtype)
        #print(rands.dtype)
        
        #print('Position')
        self.p_pos_x: pk.View1D[pk.float] = p_pos_x
        self.p_pos_y: pk.View1D[pk.float] = p_pos_y
        self.p_pos_z: pk.View1D[pk.float] = p_pos_z
        
        #print('Direction')
        self.p_dir_y: pk.View1D[pk.float] = p_dir_y
        self.p_dir_z: pk.View1D[pk.float] = p_dir_z
        self.p_dir_x: pk.View1D[pk.float] = p_dir_x
        
        #print('Cells')
        self.p_mesh_cell: pk.View1D[int] = p_mesh_cell
        self.p_speed: pk.View1D[pk.float] = p_speed
        self.p_time: pk.View1D[pk.float] = p_time
        self.p_time_cell: pk.View1D[int] = p_time_cell
        
        #print('misc')
        self.dx: pk.float = dx
        self.L: pk.float = L
        self.max_mesh_index: int = max_x
        self.dt: pk.float = dt
        self.max_time: pk.float = max_time
        self.n_mesh: int = n_mesh
        
        #print(dx)
        #print(L)
        self.num_part: int = num_part
        
        self.mesh_total_xsec: pk.View1D[pk.float] = mesh_total_xsec
        
        self.p_end_trans: pk.View1D[int] = p_end_trans
        self.rands: pk.View1D[pk.float] = rands
        #print('Mesh')
        self.mesh_dist_traveled: pk.View1D[pk.float] = mesh_dist_traveled
        self.mesh_dist_traveled_squared: pk.View1D[pk.float] = mesh_dist_traveled_squared

        self.clever_out: pk.View1D[int] = clever_out
        #print('Everything')

    @pk.main
    def run(self):
        #pk.printf('In main! \n')
        pk.parallel_for(self.num_part, self.advanceCycle_wu)
    
    @pk.workunit
    def advanceCycle_wu(self, i: int):
        
        kicker: pk.float = 1e-8
        p_dist_traveled: pk.float = 0.0
        dist_B: pk.float = 0
        
        if (self.p_end_trans[i] == 0):
            if (self.p_pos_x[i] < 0): #exited lhs
                self.p_end_trans[i] = 1
                pk.atomic_fetch_add(self.clever_out, [0], 1)
            elif (self.p_pos_x[i] >= self.L): #exited rhs
                self.p_end_trans[i] = 1
                pk.atomic_fetch_add(self.clever_out, [0], 1)
            elif (self.p_time[i] >= self.max_time):
                self.p_end_trans[i] = 1
                pk.atomic_fetch_add(self.clever_out, [0], 1)
            else:
                dist: pk.float = -math.log(self.rands[i]) / self.mesh_total_xsec[self.p_mesh_cell[i]]
                
                #x_loc: pk.float = (self.p_dir_x[i] * dist) + self.p_pos_x[i]
                LB: pk.float = self.p_mesh_cell[i] * self.dx
                RB: pk.float = LB + self.dx
                TB: pk.float = (self.p_time_cell[i]+1)*dt - self.p_time[i]
                dist_TB: pk.float = TB * p_speed[i] + kicker
                
                space_cell_inc: int = 0
                if (self.p_dir_x[i] <= 0):
                    dist_B = ((LB - self.p_pos_x[i])/self.p_dir_x[i]) + kicker
                    space_cell_inc = -1
                 #   if dist < 0:
                 #       pk.printf('Going left produced a negaitive value!\n')
                elif (self.p_dir_x[i] > 0):
                    dist_B = ((RB - self.p_pos_x[i])/self.p_dir_x[i]) + kicker
                    space_cell_inc = 1
                    #if dist < 0:
                    #    pk.printf('Going right produced a negaitive value!\n')
                    #    pk.printf('i:   %i, RB:   %f, pos_x:   %f, dir_x:  %f, cell:   %i, dist:   %f\n', i, RB, p_pos_x[i], p_dir_x[i], p_mesh_cell[i], dist_B)
                #else:
                #    pk.printf('BIG OL ERROR!')
                
                #pk.printf('\n')
                #pk.printf('Time Bound   %f\n', dist_TB)
                #pk.printf('Sampeled     %f\n', dist)
                #pk.printf('space bound: %f\n', dist_B)
                
                #pk.printf('selected:   %f\n', p_dist_traveled)
                
                #p_dist_traveled = min(dist_TB, dist_B, dist)
                cell_next: int = 0
                if dist_B < dist_TB and dist_B < dist:      #move partilce into cell
                    p_dist_traveled = dist_B
                    cell_next = self.p_mesh_cell[i] + space_cell_inc
                
                elif dist < dist_TB and dist < dist_B: #move particle in cell in time step
                    p_dist_traveled = dist
                    self.p_end_trans[i] = 1
                    cell_next = self.p_mesh_cell[i]
                    pk.atomic_fetch_add(self.clever_out, [0], 1)
                    #temp: pk.float()
                
                elif dist_TB < dist_B and dist_TB < dist:
                    p_dist_traveled = dist_TB
                    cell_next = self.p_mesh_cell[i]
                
                self.p_pos_x[i] += p_dir_x[i]*p_dist_traveled
                self.p_pos_y[i] += p_dir_x[i]*p_dist_traveled
                self.p_pos_z[i] += p_dir_x[i]*p_dist_traveled
                
                mesh_cell: int = self.p_mesh_cell[i] + (self.p_time_cell[i] * self.n_mesh)
                #if mesh_cell > 81*20:
                #    pk.printf('Mesh cell index error')
                pk.atomic_fetch_add(self.mesh_dist_traveled, [mesh_cell], p_dist_traveled)
                pk.atomic_fetch_add(self.mesh_dist_traveled_squared, [mesh_cell], p_dist_traveled**2)
                
                self.p_mesh_cell[i] = int(self.p_pos_x[i]/0.49382716049382713)
                self.p_time[i]  += dist/self.p_speed[i]
                self.p_time_cell[i] = int(self.p_time[i]/self.dt)
                
                #pk.printf('%d:  x pos after step:     %f       should be: %f\n', i, p_pos_x[i], (temp_x))
                


#@profile
def Advance(p_pos_x, p_pos_y, p_pos_z, p_mesh_cell, dx, dt, p_dir_y, p_dir_z, p_dir_x, p_speed, p_time, p_time_cell,
            num_part, mesh_total_xsec, mesh_dist_traveled, mesh_dist_traveled_squared, L, max_time):
    max_mesh_index = int(len(mesh_total_xsec)-1)
    
    p_end_trans: pk.View1D[int] = pk.View([num_part], int)#, space=pk.MemorySpace.CudaSpace) #flag
    p_end_trans.fill(0)
    
    clever_out: pk.View1D[int] = pk.View([4], int)#, space=pk.MemorySpace.CudaSpace)
    clever_out.fill(0)
    
    end_flag = 0
    cycle_count = 0
    
    n_space: int = 81
    
    while end_flag == 0:
        #allocate randoms
        summer = 0
        rands_np = np.random.random([num_part]).astype(np.float32)
        rands: pk.View1D[pk.float] = pk.View([num_part], pk.float)#, space=pk.MemorySpace.CudaSpace)
        rands[:] = rands_np[:]
        #vector of indicies for particle transport
        
        #p = pk.RangePolicy(pk.get_default_space(), 0, num_part)
        
        pre_p_mesh = p_mesh_cell
        L = float(L)
        
        space = pk.ExecutionSpace.OpenMP #pk.ExecutionSpace.OpenMP,   pk.ExecutionSpace.Cuda
        #print('*******ENTERING ADVANCE*********')
        pk.execute(space, Advance_cycle(num_part, p_pos_x, p_pos_y, p_pos_z, p_dir_y, p_dir_z, p_dir_x, p_mesh_cell, p_speed, p_time, p_time_cell, dx, dt, n_space, mesh_total_xsec, L, p_end_trans, rands, mesh_dist_traveled, mesh_dist_traveled_squared, max_mesh_index, max_time, clever_out))#pk for number still in transport
        
        summer = clever_out[0]
        
        if (summer == num_part):
            end_flag = 1
        
        
        #print(cycle_count)
        if (cycle_count > int(1e3)):
            print("************ERROR**********")
            print(" Max itter hit")
            print(p_end_trans)
            print()
            print()
            return()
        cycle_count += 1
        
    print()
    print(cycle_count)
    print(clever_out[0])
    print(clever_out[1])
    print(clever_out[2])
    print(clever_out[3])
    print()
        
        #print("Advance Complete:......{0}%       ({1}/{2})    cycle: {3}".format(int(100*summer/num_part), summer, num_part, cycle_count), end = "\r")
    #print()
    

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
