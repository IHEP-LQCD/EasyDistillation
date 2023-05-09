from itertools import product
from typing import List, Tuple

from ..insertion import InsertionRow, Operator

def get_mom2_list(mom2: int) -> List[Tuple]:
    ret = []
    for i in product(range(-3, 4), repeat=3):
        px, py, pz = i
        if px**2 + py**2 + pz**2 == mom2:
            # print(F"mom: {i}")
            ret.append((px, py, pz))
    return ret


def get_AB_opratorList_back2back(insertion_row_A: InsertionRow, insertion_row_B: InsertionRow, mom_list:List[Tuple]) -> List[Operator]:
    ops_A = []
    ops_B = []
    for i in mom_list:
        px, py, pz = i
        ops_A.append(Operator("",[insertion_row_A(px, py, pz)], [1]))
        ops_B.append(Operator("",[insertion_row_B(-px, -py, -pz)], [1]))
    return ops_A, ops_B