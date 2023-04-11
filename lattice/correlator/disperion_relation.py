from typing import Iterable, List

from ..insertion import Operator, InsertionRow
from ..filedata.abstract import FileData

from itertools import product
from .one_particle import twopoint


def getMom2Oprator(insertionRow: InsertionRow, mom2: int) -> List[Operator]:
    ret = []
    for i in product(range(-3, 4), repeat=3):
        px, py, pz = i
        if px**2 + py**2 + pz**2 == mom2:
            print(F"add mom: {i}")
            ret.append(insertionRow(px, py, pz))
    return Operator(F"mom{mom2}", ret, [1] * len(ret))


def twopointMom2(
    insertionRow: InsertionRow, mom2: int, elemental: FileData, perambulator: FileData, timeslices: Iterable[int],
    Lt: int
):
    operators = getMom2Oprator(insertionRow, mom2)
    return twopoint([operators], elemental, perambulator, timeslices, Lt)
