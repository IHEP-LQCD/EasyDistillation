import os

os.environ["CUPY_ACCELERATORS"] = "cub,cutensor"
os.environ["CUTENSOR_PATH"] = "/dg_hpc/LQCD/jiangxiangyu/libcutensor-local-repo-rhel7-1.7.0/usr"

from lattice import set_backend, get_backend

set_backend("cupy")

###############################################################################
from lattice.insertion.mom_dict import momDict_mom9
from lattice.insertion import Insertion, Operator, GammaName, DerivativeName, ProjectionName

pi_A1 = Insertion(GammaName.PI, DerivativeName.IDEN, ProjectionName.A1, momDict_mom9)
print(pi_A1[0])
op_pi = Operator("pi", [pi_A1[0](0, 0, 0)], [1])

pi2_A1 = Insertion(GammaName.PI_2, DerivativeName.IDEN, ProjectionName.A1, momDict_mom9)
print(pi2_A1[0])
op_pi2 = Operator("pi2", [pi2_A1[0](0, 0, 0)], [1])

rho_T1 = Insertion(GammaName.RHO, DerivativeName.IDEN, ProjectionName.T1, momDict_mom9)
print(rho_T1[0])
op_rho = Operator("rho", [rho_T1[0](0, 0, 0)], [1])
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

###############################################################################

###############################################################################
from lattice import QuarkDiagram, compute_diagrams_multitime, Meson, Propagator, PropagatorLocal

for cfg in ["2000"]:
    e = elemental.load(cfg)
    p = perambulator.load(cfg)
    connected = QuarkDiagram([[0, 1], [1, 0]])
    disconnected = QuarkDiagram([[2, 0], [0, 2]])
    eta_src = Meson(e, op_pi2, True)
    eta_snk = Meson(e, op_pi2, False)
    propag = Propagator(p, 128)
    propag_local = PropagatorLocal(p, 128)

    import numpy
    t_snk = numpy.arange(128)
    backend = get_backend()

    twopt = backend.zeros((2, 128))
    for t_src in range(128):
        print(t_src)
        tmp = compute_diagrams_multitime(
            [connected, disconnected],
            [t_src, t_snk],
            [eta_src, eta_snk],
            [None, propag, propag_local],
        ).real
        twopt += backend.roll(tmp, -t_src, 1)
    twopt /= 128
    twopt[1] = -twopt[0] + 2 * twopt[1]
    twopt[0] = -twopt[0]
    print(twopt)
    print(backend.arccosh((backend.roll(twopt, -1, 1) + backend.roll(twopt, 1, 1)) / twopt / 2))

    rho2pipi = QuarkDiagram([[0, 1, 0], [0, 0, 2], [1, 0, 0]])
    rho_src = Meson(e, op_rho, True)
    pi_snk = Meson(e, op_pi, False)
    twopt = backend.zeros((1, 128))
    propag = Propagator(p, 128)
    propag_local = PropagatorLocal(p, 128)
    for t_src in range(128):
        print(t_src)
        tmp = compute_diagrams_multitime(
            [rho2pipi],
            [t_src, t_snk, t_snk],
            [rho_src, pi_snk, pi_snk],
            [None, propag, propag_local],
        ).real
        twopt += backend.roll(tmp, -t_src, 1)
    twopt /= 128
    print(twopt)
###############################################################################
