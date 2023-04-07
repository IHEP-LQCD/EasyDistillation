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
    Np = len(operators[0].momentum)
    Nt = len(timeslices)

    ret = numpy.zeros((Nop, Np, Nt), "<c16")
    phis = getElementalData(operators, [1], elemental)
    for t in timeslices:
        print(t)
        tau = perambulator[t]
        print(f"{perambulator.sizeInByte/perambulator.timeInSec/1024**2} MB/s")
        tau_bw = contract("ij,tkjba,kl->tilab", gamma(15), tau.conj(), gamma(15))
        for idx in range(Nop):
            phi = phis[idx]
            ret[idx] += contract(
                "tijab,xjk,xptbc,tklcd,yli,ypad->pt", tau_bw, phi[0], numpy.roll(phi[1], -t, 1), tau, phi[0],
                phi[1][:, :, t].conj()
            )
    ret /= Nt

    return ret
