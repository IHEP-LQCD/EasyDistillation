from typing import Iterable, List
from opt_einsum import contract

from ..insertion import Operator
from ..insertion.gamma import gamma
from ..data import getElementalData
from ..filedata.abstract import FileData
from ..backend import getBackend


def twopoint(
    operators: List[Operator], elemental: FileData, perambulator: FileData, timeslices: Iterable[int], Lt: int
):
    numpy = getBackend()
    Nop = len(operators)
    Nt = len(timeslices)

    ret = numpy.zeros((Nop, Lt), "<c16")
    phis = getElementalData(operators, elemental)
    for t in timeslices:
        tau = perambulator[t]
        tau_bw = contract("ii,tjiba,jj->tijab", gamma(15), tau.conj(), gamma(15))
        for idx in range(Nop):
            phi = phis[idx]
            ret[idx] += contract(
                "tijab,xjk,xtbc,tklcd,yli,yad->t", tau_bw, phi[0], numpy.roll(phi[1], -t, 1), tau, phi[0],
                phi[1][:, t].conj()
            )
        print(f"t{t}: {perambulator.sizeInByte/perambulator.timeInSec/1024**2} MB/s")
    ret /= Nt

    return -ret


def twopointMatrix(
    operators: List[Operator], elemental: FileData, perambulator: FileData, timeslices: Iterable[int], Lt: int
):
    numpy = getBackend()
    Nop = len(operators)
    Nt = len(timeslices)

    ret = numpy.zeros((Nop, Nop, Lt), "<c16")
    phis = getElementalData(operators, elemental)
    for t in timeslices:
        tau = perambulator[t]
        tau_bw = contract("ii,tjiba,jj->tijab", gamma(15), tau.conj(), gamma(15))
        for isrc in range(Nop):
            for isnk in range(Nop):
                phi_src = phis[isrc]
                gamma_src = contract("ij,xkj,kl->xil", gamma(8), phi_src[0].conj(), gamma(8))
                phi_snk = phis[isnk]
                ret[isrc, isnk] += contract(
                    "tijab,xjk,xtbc,tklcd,yli,yad->t", tau_bw, phi_snk[0], numpy.roll(phi_snk[1], -t, 1), tau,
                    gamma_src, phi_src[1][:, t].conj()
                )
        print(f"t{t}: {perambulator.sizeInByte/perambulator.timeInSec/1024**2} MB/s")
    ret /= Nt
    return -ret


def twopointIsoscalar(
    operators: List[Operator], elemental: FileData, perambulator: FileData, timeslices: Iterable[int], Lt: int
):
    numpy = getBackend()
    Nop = len(operators)
    Nt = len(timeslices)
    if Lt != Nt:
        raise ValueError(F"Disconnect must compute full timeslices!")

    connected = numpy.zeros((Nop, Lt), "<c16")
    loop_src = numpy.zeros((Nop, Lt), "<c16")
    loop_snk = numpy.zeros((Nop, Lt), "<c16")
    phis = getElementalData(operators, elemental)

    for t in timeslices:
        tau = perambulator[t]
        tau_bw = contract("ii,tjiba,jj->tijab", gamma(15), tau.conj(), gamma(15))
        for idx in range(Nop):
            phi = phis[idx]
            gamma_src = contract("ij,xkj,kl->xil", gamma(8), phi[0].conj(), gamma(8))
            connected[idx] += contract(
                "tijab,xjk,xtbc,tklcd,yli,yad->t", tau_bw, phi[0], numpy.roll(phi[1], -t, 1), tau, gamma_src,
                phi[1][:, t].conj()
            )
            loop_src[idx, t] = contract("ijab,yji,yab", tau[0], gamma_src, phi[1][:, t].conj())
            loop_snk[idx, t] = contract("ijab,xji,xba", tau[0], phi[0], phi[1][:, t])
        print(f"t{t}: {perambulator.sizeInByte/perambulator.timeInSec/1024**2} MB/s")
    connected /= Nt

    disconnected = contract("xi, xj -> xij", loop_src, loop_snk)
    for t in timeslices:
        disconnected[:,t,:] = numpy.roll(disconnected[:,t,:], -t, axis=1)
    disconnected = disconnected.mean(1)
    return -connected + 2 * disconnected