from time import perf_counter

class mytimer:
    def __init__(self, info: str, size: int = -1) -> None:
        self.info = info
        self.size = size
        print(f"TM_BEGIN: {self.info}.")
        self.T_begin = perf_counter()
        self.mark0 = perf_counter()
        self.dtype_size = 16

    def __call__(self):
        self.mark1 = perf_counter()
        print(f"{self.info}, time = {(self.mark1-self.mark0):.5f} s.")
        self.mark0 = self.mark1

    def end(self):
        self.T_end = perf_counter()
        if self.size != -1:
            print(
                f"TM_END: {self.info}, time = {(self.T_end-self.T_begin):.5f} s, speed = {self.size*self.dtype_size/(self.T_end-self.T_begin)/1024**2:.5f} Mb/s.")
        else:
            print(f"TM_END: {self.info}, time = {(self.T_end-self.T_begin):.5f} s.")