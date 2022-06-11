import pykokkos as pk




@pk.workload
class StillInSpace:
    def __init__(self, p_pos_x, surface_distances, p_alive, num_part, clever_out):
        self.p_alive: pk.View1D[int] = p_alive
        self.num_part: int = num_part
        self.surface_distances: pk.View1D[pk.float] = surface_distances
        self.p_pos_x: pk.View1D[pk.float] = p_pos_x
        self.clever_out: pk.Veiw1D[int] = clever_out

    @pk.main
    def run(self):
        tally_left: pk.int = 0
        tally_right: pk.int = 0
        for i in range(self.num_part):
            #exit at left
            if p_pos_x[i] <= self.surface_distances[0]:
                tally_left += 1
                self.p_alive[i] = False
                
            elif p_pos_x[i] >= self.surface_distances[len(self.surface_distances)-1]:
                tally_right += 1
                self.p_alive[i] = False
            
        clever_out[0] = tally_left
        clever_out[1] = tally_right


@pk.workload
class StillInTime:
    def __init__(self, p_time, max_time, p_alive, num_part, clever_out):
        self.p_time: pk.View1D[pk.float] = p_time
        self.p_alive: pk.View1D[int] = p_alive
        self.max_time: int = max_time
        self.num_part: int = num_part
        self.clever_out: pk.View1D[int] = clever_out

    @pk.main
    def run(self):
        tally_time: int = 0
    
        for i in range(num_part):
            if self.p_time[i] >= self.max_time:
                self.p_alive[i] = 0
                tally_time += 1
            
        self.clever_out[2] = tally_time
