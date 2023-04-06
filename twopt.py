from typing import Dict, List
from opt_einsum import contract

from lattice.filedata.abstract import FileData
from lattice.backend import setBackend, getBackend
from lattice import preset, insertion
from lattice.data import getElementalData
from lattice.insertion import Insertion, Operator
from lattice.insertion.gamma import gamma
from lattice.insertion.mom_dict import momDict_mom9

import cupy
setBackend(cupy)
cupy.cuda.Device(1).use()


def calcTwopt(operators: List[Operator], elemental: FileData, perambulator: FileData):
    numpy = getBackend()
    Nop = len(operators)

    ret = numpy.zeros((Nop, 1, 128), "<c16")
    phis = getElementalData(operators, [1], elemental)
    for t in range(128):
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
    ret /= 128

    return ret


elemental = preset.ElementalNpy(
    "/dg_hpc/LQCD/shichunjiang/DATA/clqcd_nf2_clov_L16_T128_b2.0_ml-0.05862_sn2_srho0.12_gg5.65_gf5.2_usg0.780268_usf0.949104/04.meson.mom9/clqcd_nf2_clov_L16_T128_b2.0_xi5_ml-0.05862_cfg_",
    ".mom9.npy", [4, 123, 128, 70, 70], 70
)
perambulator = preset.PerambulatorNpy(
    "/dg_hpc/LQCD/shichunjiang/DATA/clqcd_nf2_clov_L16_T128_b2.0_ml-0.05862_sn2_srho0.12_gg5.65_gf5.2_usg0.780268_usf0.949104/03.perambulator.light.single.prec1e-9/clqcd_nf2_clov_L16_T128_b2.0_xi5_ml-0.05862_cfg_",
    ".peram.npy", [128, 128, 4, 4, 70, 70], 70
)
insertionField = Insertion(
    insertion.GAMMA_NAME.PI, insertion.DERIVATIVE_NAME.IDEN, insertion.PROJECTION_NAME.A1, momDict_mom9
)
print(insertionField)

e = elemental.load("2000")
p = perambulator.load("2000")
twopt = calcTwopt([insertionField[0](0, 0, 0)], e, p).reshape(128)
numpy = getBackend()
print(numpy.arccosh((numpy.roll(twopt, -1, 0) + numpy.roll(twopt, 1, 0)) / twopt / 2))
