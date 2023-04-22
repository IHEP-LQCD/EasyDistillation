from lattice import set_backend, get_backend

set_backend("cupy")

###############################################################################
from lattice.insertion.mom_dict import momDict_mom9
from lattice.insertion import Insertion, Operator, GammaName, DerivativeName, ProjectionName

pi_A1 = Insertion(GammaName.PI, DerivativeName.IDEN, ProjectionName.A1, momDict_mom9)
print(pi_A1[0])
op_pi = Operator("pi", [pi_A1[0](0, 0, 0)], [1])

b1xnabla_A1 = Insertion(GammaName.B1, DerivativeName.NABLA, ProjectionName.A1, momDict_mom9)
print(b1xnabla_A1[0])
op_pi2 = Operator("pi2", [pi_A1[0](0, 0, 0), b1xnabla_A1[0](0, 0, 0)], [3, 1])
###############################################################################

###############################################################################
from lattice import preset

elemental = preset.ElementalNpy(
    "/dg_hpc/LQCD/shichunjiang/DATA/clqcd_nf2_clov_L16_T128_b2.0_ml-0.05862_sn2_srho0.12_gg5.65_gf5.2_usg0.780268_usf0.949104/04.meson.mom9/clqcd_nf2_clov_L16_T128_b2.0_xi5_ml-0.05862_cfg_",
    ".mom9.npy", [4, 123, 128, 70, 70], 70
)
perambulator = preset.PerambulatorNpy(
    "/dg_hpc/LQCD/shichunjiang/DATA/clqcd_nf2_clov_L16_T128_b2.0_ml-0.05862_sn2_srho0.12_gg5.65_gf5.2_usg0.780268_usf0.949104/03.perambulator.light.single.prec1e-9/clqcd_nf2_clov_L16_T128_b2.0_xi5_ml-0.05862_cfg_",
    ".peram.npy", [128, 128, 4, 4, 70, 70], 70
)

cfg = "2000"
e = elemental.load(cfg)
p = perambulator.load(cfg)
###############################################################################

###############################################################################
from lattice.correlator.one_particle import twopoint, twopoint_matrix

backend = get_backend()

# compute 2pt
twopt = twopoint([op_pi, op_pi2], e, p, list(range(128)), 128)  # [Nop, Lt]
twopt = twopt.real
print(backend.arccosh((backend.roll(twopt, -1, 1) + backend.roll(twopt, 1, 1)) / twopt / 2))

# compute a 2 by 2 two-point correlation matrix
twopt_matrix = twopoint_matrix([op_pi, op_pi2], e, p, list(range(128)), 128)
twopt_matrix = twopt_matrix.real
print(
    backend.arccosh(
        (backend.roll(twopt_matrix[0, 0], -1, 0) + backend.roll(twopt_matrix[0, 0], 1, 0)) / twopt_matrix[0, 0] / 2
    )
)
print(
    backend.arccosh(
        (backend.roll(twopt_matrix[1, 1], -1, 0) + backend.roll(twopt_matrix[1, 1], 1, 0)) / twopt_matrix[1, 1] / 2
    )
)
###############################################################################

# compute summation of p2 = 1 2pt
from lattice.correlator.disperion_relation import twopoint_mom2

twopt_mom2 = twopoint_mom2(pi_A1[0], 2, e, p, list(range(128)), 128)
print(twopt_mom2)
