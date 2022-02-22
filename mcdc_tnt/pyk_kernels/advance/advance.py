import pykokkos as pk
import numpy.random as rand
import math


@pk.workunit
def advance (i: int,
            p_pos_x: pk.View1D[pk.double], p_pos_y: pk.View1D[pk.double], p_pos_z: pk.View1D[pk.double],
            i_h: pk.View1D[pk.double], j_h: pk.View1D[pk.double], k_h: pk.View1D[pk.double], 
            xi: pk.View1D[pk.double]):
            
    mesh_total_xsec: float = 3.0154
    
    dist: float = -math.log(xi[i]) / mesh_total_xsec
    
    p_pos_x[i] += i_h[i]*dist
    p_pos_y[i] += j_h[i]*dist
    p_pos_z[i] += k_h[i]*dist
    
    
n = int(1e8)
    
space = pk.ExecutionSpace.OpenMP
pk.set_default_space(space)

i_h: pk.View1D[pk.double] = pk.View([n], pk.double)
j_h: pk.View1D[pk.double] = pk.View([n], pk.double)
k_h: pk.View1D[pk.double] = pk.View([n], pk.double)
p_pos_x: pk.View1D[pk.double] = pk.View([n], pk.double)
p_pos_x.fill(2.31)
p_pos_y: pk.View1D[pk.double] = pk.View([n], pk.double)
p_pos_x.fill(2.31)
p_pos_z: pk.View1D[pk.double] = pk.View([n], pk.double)
p_pos_x.fill(2.31)

xi: pk.View1D[pk.double] = pk.View([n], pk.double)

for i in range(n):
    xi[i] = rand.random()
    i_h[i] = rand.random()
    j_h[i] = rand.random()
    k_h[i] = rand.random()
    
    
print("RANDOM NUMBERS ALLOCATED")

p = pk.RangePolicy(pk.get_default_space(), 0, n)

timer = pk.Timer()
pk.parallel_for(n, advance, p_pos_x=p_pos_x, p_pos_y=p_pos_y, p_pos_z=p_pos_z, i_h=i_h, j_h=j_h, k_h=k_h, xi=xi)
timer_result = timer.seconds()

print(timer_result)
