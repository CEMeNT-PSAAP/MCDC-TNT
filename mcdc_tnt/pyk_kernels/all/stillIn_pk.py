import pykokkos as pk




@pk.workload
class StillInSpace:
    def __init__(self, p_pos_x, L, p_alive, num_part, clever_out):
        self.p_alive: pk.View1D[int] = p_alive
        self.num_part: int = num_part
        self.L: pk.float = 40
        self.p_pos_x: pk.View1D[pk.float] = p_pos_x
        self.clever_out: pk.View1D[int] = clever_out
        
    @pk.main
    def run(self):
        tally_left: int = 0
        tally_right: int = 0
        for i in range(self.num_part):
            #exit at left
            if p_pos_x[i] <= 0:
                tally_left += 1
                self.p_alive[i] = False
                
            elif p_pos_x[i] >= self.L:
                tally_right += 1
                self.p_alive[i] = False
            
        self.clever_out[0] = tally_left
        self.clever_out[1] = tally_right


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
