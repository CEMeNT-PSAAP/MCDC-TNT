import pykokkos as pk
import numpy.random as rand
import math
import pykokkos as pk
import random

rand.seed(777)

@pk.workunit
def add(i: int, i_h: pk.View1D[pk.double], j_h: pk.View1D[pk.double], k_h: pk.View1D[pk.double], xi: pk.View1D[pk.double]):
    mu: float = 2.0*xi[i] - 1.0
    azi: float = 2.0*xi[i] - 1.0
    c: float = (1.0 - mu**2)**0.5
    
    i_h[i] = math.cos(azi) * c
    j_h[i] = math.sin(azi) * c
    k_h[i] = mu

n = int(1e5)

space = pk.ExecutionSpace.OpenMP
pk.set_default_space(space)

i_h: pk.View1D[pk.double] = pk.View([n], pk.double)
j_h: pk.View1D[pk.double] = pk.View([n], pk.double)
k_h: pk.View1D[pk.double] = pk.View([n], pk.double)
xi: pk.View1D[pk.double] = pk.View([2*n], pk.double)

print("VIEWS ALLOCATED")

for i in range(n):
    i_h[i] = rand.random()
    j_h[i] = rand.random()
    k_h[i] = rand.random()

print("RANDOM NUMBERS ALLOCATED")

for i in range(int(2*n)):
    xi[i] = rand.random()

print("RANDOM NUMBERS ALLOCATED")

p = pk.RangePolicy(pk.get_default_space(), 0, n)

print("Entering Run")

timer = pk.Timer()
pk.parallel_for(n, add, i_h=i_h, j_h=j_h, k_h=k_h, xi=xi)
timer_result = timer.seconds()

print(timer_result)

#print(i_h)

#print(j_h)
#print(k_h)
