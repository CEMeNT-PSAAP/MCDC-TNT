import numpy as np
import numba.cuda as cuda


@cuda.reduce
def red(a, b):
    return a + b
    
    
    
n = 1000000
a = np.random.random(n).astype(np.float64)

got = red(a)
exp = a.sum()

print(got)
print(exp)

assert exp == got

