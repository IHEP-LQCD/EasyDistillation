from typing import Iterable, List
from opt_einsum import contract

from ..insertion import Operator
from ..insertion.gamma import gamma
from ..data import getElementalData
from ..filedata.abstract import FileData
from ..backend import getBackend


def twopoint(operators: List[Operator], elemental: FileData, perambulator: FileData, timeslices: Iterable[int]):
    numpy = getBackend()
    Nop = len(operators)
    Nt = len(timeslices)

    ret = numpy.zeros((Nop, Nt), "<c16")
    phis = getElementalData(operators, elemental)
    for t in timeslices:
        tau = perambulator[t]
        tau_bw = contract("ij,tkjba,kl->tilab", gamma(15), tau.conj(), gamma(15))
        for idx in range(Nop):
            phi = phis[idx]
            ret[idx] += contract(
                "tijab,xjk,xtbc,tklcd,yli,yad->t", tau_bw, phi[0], numpy.roll(phi[1], -t, 1), tau, phi[0],
                phi[1][:, t].conj()
            )
        print(f"{perambulator.sizeInByte/perambulator.timeInSec/1024**2} MB/s")
    ret /= Nt

    return ret
