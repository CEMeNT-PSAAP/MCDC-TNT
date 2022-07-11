import numba as nb
import numpy as np

@nb.jit(nopython=True) 
def StillInSpace(p_pos_x, surface_distances, p_alive, num_part):
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


@nb.jit(nopython=True) 
def StillInTime(p_time, max_time, p_alive, num_part):
    
    tally_time: int = 0
    
    for i in range(num_part):
        if p_time[i] >= max_time:
            p_alive[i] = 0
            tally_time +=1
            
    return(p_alive, tally_time)
