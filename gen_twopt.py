import cupy

from lattice.backend import setBackend, getBackend
setBackend(cupy)
# cupy.cuda.Device(1).use()
from lattice import preset, Dispatch
from lattice.insertion import Insertion, Operator, GAMMA_NAME, DERIVATIVE_NAME, PROJECTION_NAME
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
insertionField = Insertion(GAMMA_NAME.PI, DERIVATIVE_NAME.IDEN, PROJECTION_NAME.A1, momDict_mom9)
print(insertionField[0])
operator_pi = Operator("pi", [insertionField[0](0, 0, 0)], [1])

insertionField2 = Insertion(GAMMA_NAME.B1, DERIVATIVE_NAME.NABLA, PROJECTION_NAME.A1, momDict_mom9)
print(insertionField2[0])
operator_pi2 = Operator("pi2", [insertionField2[0](0, 0, 0)], [1])

np = getBackend()
twopt = np.zeros((2, 128), "<c16")
dispatcher = Dispatch("cfglist.full.txt", "aaa")
for cfg in dispatcher:
    e = elemental.load(cfg)
    p = perambulator.load(cfg)
    # compute 2pt
    twopt += twopoint([operator_pi, operator_pi2], e, p, list(range(128)), 128)
twopt = twopt.real
print(np.arccosh((np.roll(twopt, -1, 1) + np.roll(twopt, 1, 1)) / twopt / 2))

# compute a 2 by 2 two-point correlation matrix
twoptMatrix = twopointMatrix([operator_pi, operator_pi2], e, p, list(range(128)), 128)
twoptMatrix = twoptMatrix.real
print(np.arccosh((np.roll(twoptMatrix[0, 0], -1, 0) + np.roll(twoptMatrix[0, 0], 1, 0)) / twoptMatrix[0, 0] / 2))
print(np.arccosh((np.roll(twoptMatrix[1, 1], -1, 0) + np.roll(twoptMatrix[1, 1], 1, 0)) / twoptMatrix[1, 1] / 2))

# compute summation of p2 = 1 2pt
from lattice.correlator.disperion_relation import twopointMom2
twoptMom2 = twopointMom2(insertionField[0], 2, e, p, list(range(128)), 128)
print(twoptMom2)
