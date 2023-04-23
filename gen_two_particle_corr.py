from lattice.backend import set_backend, get_backend

set_backend("cupy")

from lattice import Dispatch, preset
from lattice.insertion.mom_dict import momDict_mom9
from lattice.insertion import Insertion, Operator, GammaName, DerivativeName, ProjectionName

from lattice import Meson, Propagator, PropagatorLocal, QuarkDiagram, compute_diagrams_multitime

Lt = 128

ins_D = Insertion(GammaName.PI, DerivativeName.IDEN, ProjectionName.A1, momDict_mom9)
ins_Dstar = Insertion(GammaName.RHO, DerivativeName.IDEN, ProjectionName.T1, momDict_mom9)
ins_chic1 = Insertion(GammaName.A1, DerivativeName.IDEN, ProjectionName.T1, momDict_mom9)
op_chic1 = Operator("Chic1", [ins_chic1[0](0, 0, 0)], [1])

# op_A = u_bar gamma5 c
op_D = Operator("D", [ins_D[0](0, 0, 0)], [1])
# op_B = c_bar gamma_i u
op_Ds = Operator("Dbar_star", [ins_Dstar[2](0, 0, 0)], [1])

# Read peramulators and elemental from file
elemental = preset.ElementalNpy(
    "/dg_hpc/LQCD/jiangxiangyu/chkDeriv/DATA/light.20200720.b20.16_128/04.meson/", ".meson.npy",
    [128, 13, 123, 70, 70], 70
)
perambulator_light = preset.PerambulatorBinary(
    "/dg_hpc/LQCD/DATA/light.20200720.b20.16_128/03.perambulator/", ".peram", [128, 128, 4, 4, 70, 70], 70
)
perambulator_charm = preset.PerambulatorBinary(
    "/dg_hpc/LQCD/DATA/light.20200720.b20.16_128/03.perambulator.charm/", ".charm.peram", [128, 128, 4, 4, 70, 70], 70
)

D_D = QuarkDiagram([
    [0, 0, 1, 0],
    [0, 0, 0, 0],
    [2, 0, 0, 0],
    [0, 0, 0, 0],
])
Ds_Ds = QuarkDiagram([
    [0, 0, 0, 0],
    [0, 0, 0, 2],
    [0, 0, 0, 0],
    [0, 1, 0, 0],
])
DDsbar_DDsbar_direct = QuarkDiagram([
    [0, 0, 1, 0],
    [0, 0, 0, 2],
    [2, 0, 0, 0],
    [0, 1, 0, 0],
])
DDsbar_DDsbar_cross = QuarkDiagram([
    [0, 0, 2, 0],
    [3, 0, 0, 0],
    [0, 0, 0, 3],
    [0, 2, 0, 0],
])
chic1_DDs = QuarkDiagram([
    [0, 0, 1, 0],
    [0, 0, 0, 3],
    [1, 0, 0, 0],
    [0, 0, 0, 0],
])
chic1_ccbar_eta = QuarkDiagram([
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 3],
    [0, 0, 0, 0],
])

line_light = Propagator(perambulator_light, Lt)
line_charm = Propagator(perambulator_charm, Lt)
line_local_light = PropagatorLocal(perambulator_light, Lt)
D_src = Meson(elemental, op_D, True)
D_snk = Meson(elemental, op_D, False)
Ds_src = Meson(elemental, op_Ds, True)
Ds_snk = Meson(elemental, op_Ds, False)

# cfg = "s1.0_cfg_2000.stout.n20.f0.12.nev70"
dispatcher = Dispatch("cfglist.700.txt", "balabala")
for cfg in dispatcher:
    line_light.load(cfg)
    line_charm.load(cfg)
    line_local_light.load(cfg)
    D_src.load(cfg)
    D_snk.load(cfg)
    Ds_src.load(cfg)
    Ds_snk.load(cfg)

    import numpy
    t_snk = numpy.arange(128)
    backend = get_backend()

    twopt = backend.zeros((6, 128), "<c16")
    for t_src in range(128):
        print(t_src)
        tmp = compute_diagrams_multitime(
            [D_D, Ds_Ds, DDsbar_DDsbar_direct, DDsbar_DDsbar_cross, chic1_DDs, chic1_ccbar_eta],
            [t_src, t_src, t_snk, t_src],
            [D_src, Ds_src, D_snk, Ds_snk],
            [None, line_charm, line_light, line_local_light],
        )
        twopt[0:6] += backend.roll(tmp, -t_src, 1)[0:6]

    twopt /= 128
    print(twopt)
    print(backend.arccosh((backend.roll(twopt, -1, 1) + backend.roll(twopt, 1, 1)) / twopt / 2))
