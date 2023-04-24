from typing import Iterable, List

from ..insertion import Operator, InsertionRow
from ..filedata.abstract import FileData

from itertools import product
from .one_particle import twopoint


def get_mom2_oprator(insertion_row: InsertionRow, mom2: int) -> Operator:
    ret = []
    for i in product(range(-3, 4), repeat=3):
        px, py, pz = i
        if px**2 + py**2 + pz**2 == mom2:
            print(F"add mom: {i}")
            ret.append(insertion_row(px, py, pz))
    return Operator(F"mom{mom2}", ret, [len(ret)**-.5] * len(ret))


def twopoint_mom2(
    insertion_row: InsertionRow, mom2: int, elemental: FileData, perambulator: FileData, timeslices: Iterable[int],
    Lt: int
):
    operators = get_mom2_oprator(insertion_row, mom2)
    return twopoint([operators], elemental, perambulator, timeslices, Lt)
