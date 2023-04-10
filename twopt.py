import cupy

from lattice.backend import setBackend, getBackend
setBackend(cupy)
cupy.cuda.Device(1).use()
from lattice import preset, insertion
from lattice.insertion import Insertion, Operator
from lattice.insertion.mom_dict import momDict_mom9
from lattice.correlator.one_particle import twopoint, twopointMatrix

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
operator_pi = Operator("pi", [insertionField[0](0, 0, 0)], [1])
print(operator_pi.name, operator_pi.parts)

e = elemental.load("2000")
p = perambulator.load("2000")
numpy = getBackend()
# compute 2pt
twopt = twopoint([operator_pi], e, p, list(range(128)), 128).reshape(128)
print(numpy.arccosh((numpy.roll(twopt, -1, 0) + numpy.roll(twopt, 1, 0)) / twopt / 2))

# compute a 2 by 2 two-point correlation matrix
insertionField2 = Insertion(
    insertion.GAMMA_NAME.B1, insertion.DERIVATIVE_NAME.NABLA, insertion.PROJECTION_NAME.A1, momDict_mom9
)
operator_pi2 = Operator("pi2", [insertionField2[0](0, 0, 0)], [1])
print(operator_pi2.name, operator_pi2.parts)
twoptMatrix = twopointMatrix([operator_pi, operator_pi2], e, p, list(range(128)[::4]), 128)
print(numpy.arccosh((numpy.roll(twoptMatrix[0,0], -1, 0) + numpy.roll(twoptMatrix[0,0], 1, 0)) / twoptMatrix[0,0] / 2))
print(numpy.arccosh((numpy.roll(twoptMatrix[1,1], -1, 0) + numpy.roll(twoptMatrix[1,1], 1, 0)) / twoptMatrix[1,1] / 2))

# compute summation of p2 = 1 2pt
from lattice.correlator.disperion_relation import twopointMom2
twoptMom2 = twopointMom2(insertionField[0], 2, e, p, list(range(128)), 128)
print(twoptMom2)