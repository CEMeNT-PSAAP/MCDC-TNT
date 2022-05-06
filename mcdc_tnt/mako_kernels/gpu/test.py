import pycuda.autoinit
import pycuda.driver as drv
import numpy as np

from pycuda.compiler import SourceModule
mod = SourceModule("""
#include <stdio.h>

__global__ void multiply_them(float *dest, float *out, int *accum)
{
      const int i = threadIdx.x + blockIdx.x * blockDim.x;
      
      if (i < 5000){
          dest[i] = 1;
          out[i] = i;
          atomicAdd(&accum[0], 1);
      }
}



__global__ void say_hi()
{
  printf("I am %d.%d\\n", threadIdx.x, threadIdx.y);
}

""")

num_part: int = 5000

multiply_them = mod.get_function("multiply_them")

threadsperblock = 32
blockspergrid = (num_part + (threadsperblock - 1)) // threadsperblock

dest = np.ones(num_part, np.float32)
out = np.zeros_like(dest)

accum = np.zeros(1, np.int32)

multiply_them(
        drv.InOut(dest), drv.Out(out), drv.InOut(accum),
        block = (threadsperblock, 1, 1), grid=(blockspergrid, 1))

print(np.sum(dest)==num_part)
print(accum)
print(accum[0] == num_part)

func = mod.get_function("say_hi")
func(block=(4,4,1))
