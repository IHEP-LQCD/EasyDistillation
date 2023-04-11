import cupy

from lattice.backend import setBackend, getBackend
setBackend(cupy)
# cupy.cuda.Device(1).use()
from lattice import preset, Dispatch
from lattice.insertion import Insertion, Operator, GAMMA_NAME, DERIVATIVE_NAME, PROJECTION_NAME
from lattice.insertion.mom_dict import momDict_mom9
from lattice.correlator.one_particle import twopointIsoscalar

elemental = preset.ElementalNpy(
    "/dg_hpc/LQCD/shichunjiang/DATA/clqcd_nf2_clov_L16_T128_b2.0_ml-0.05862_sn2_srho0.12_gg5.65_gf5.2_usg0.780268_usf0.949104/04.meson.mom9/clqcd_nf2_clov_L16_T128_b2.0_xi5_ml-0.05862_cfg_",
    ".mom9.npy", [4, 123, 128, 70, 70], 70
)
perambulator = preset.PerambulatorNpy(
    "/dg_hpc/LQCD/shichunjiang/DATA/clqcd_nf2_clov_L16_T128_b2.0_ml-0.05862_sn2_srho0.12_gg5.65_gf5.2_usg0.780268_usf0.949104/03.perambulator.light.single.prec1e-9/clqcd_nf2_clov_L16_T128_b2.0_xi5_ml-0.05862_cfg_",
    ".peram.npy", [128, 128, 4, 4, 70, 70], 70
)

insertionField = Insertion(GAMMA_NAME.PI_2, DERIVATIVE_NAME.IDEN, PROJECTION_NAME.A1, momDict_mom9)
print(insertionField[0])
operator_pi = Operator("pi2", [insertionField[0](0, 0, 0)], [1])

np = getBackend()
twopt = np.zeros((1, 128), "<c16")
dispatcher = Dispatch("cfglist.txt", "aaa")
for cfg in dispatcher:
    e = elemental.load(cfg)
    p = perambulator.load(cfg)
    # compute isoscalar 2pt
    twopt += twopointIsoscalar([operator_pi], e, p, list(range(128)), 128)
print(twopt)
twopt = twopt.real
print(np.arccosh((np.roll(twopt, -1, 1) + np.roll(twopt, 1, 1)) / twopt / 2))