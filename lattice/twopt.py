from typing import Dict, List
from opt_einsum import contract

from .abstract import FileData
from .product import Operator
from .backend import getBackend
from .gamma import instance, gamma


def solve(operators: List[Operator], derivDict, elemental: FileData):
    numpy = getBackend()
    ret = []
    cached_idx = []
    cache = numpy.zeros((13, 128, 27, 70, 70), "<c16")

    for operator in operators:
        gamma = []
        deriv = []
        for operatorPart in operator.parts:
            gamma.append(instance(operatorPart.gamma))
            deriv.append(numpy.zeros((128, 27, 70, 70), "<c16"))
            for derivPart in operatorPart.deriv.parts:
                coeff = derivPart.coeff
                idx = derivDict[derivPart.deriv]
                if idx not in cached_idx:
                    cache[idx] = elemental[:, idx]
                    cached_idx.append(idx)
                deriv[-1] += coeff * cache[idx]
        ret.append((numpy.asarray(gamma), numpy.asarray(deriv)))

    return ret


def calcTwopt(operators: List[Operator], elemental: FileData, perambulator: FileData, deriv: Dict[str, int]):
    numpy = getBackend()

    # print(cupy.get_default_memory_pool().used_bytes() / 1024 ** 3)
    ret = numpy.zeros((8, 128, 27), "<c16")
    phis = solve(operators, deriv, elemental)
    for t in range(128):
        tau = perambulator[t]
        tau_bw = contract("ij,tkjba,kl->tilab", gamma(15), tau.conj(), gamma(15))
        for idx in range(len(operators)):
            phi = phis[idx]
            ret[idx] += contract("tijab,xjk,xtpbc,tklcd,yli,ypad->tp", tau_bw, phi[0], numpy.roll(phi[1], -t, 1), tau, phi[0], phi[1][:, t].conj())
    ret /= 128

    return ret
