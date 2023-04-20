import cupy as cp

from lattice.backend import set_backend, get_backend
# from lattice.data import get_elemental_data

set_backend(cp)
numpy = get_backend()

from lattice import Dispatch, preset
from lattice.insertion.mom_dict import momDict_mom9
from lattice.insertion import Insertion, Operator, GammaName, DerivativeName, ProjectionName

from lattice import Meson, Propagator, PropagatorLocal, QuarkDiagram, compute_diagrams_multitime

Nt = 128

ins_D = Insertion(GammaName.PI, DerivativeName.IDEN, ProjectionName.A1, momDict_mom9)

ins_Dstar = Insertion(GammaName.RHO, DerivativeName.IDEN, ProjectionName.T1, momDict_mom9)

ins_chic1 = Insertion(GammaName.A1, DerivativeName.IDEN, ProjectionName.T1, momDict_mom9)
op_chic1 = Operator("Chic1", [ins_chic1[0](0, 0, 0)], [1])

# op_A = u_bar gamma5 c
op_D = Operator("D", [ins_D[0](0, 0, 0)], [1])

# op_B = c_bar gamma_i u
op_Ds = Operator("Dbar_star", [ins_Dstar[2](0, 0, 0)], [1])

# Read peramulators and elemental from file
elemental_pre = preset.ElementalNpy(
    "/dg_hpc/LQCD/jiangxiangyu/chkDeriv/DATA/light.20200720.b20.16_128/04.meson/", ".meson.npy",
    [128, 13, 123, 70, 70], 70
)
perambulator_light_pre = preset.PerambulatorBinary(
    "/dg_hpc/LQCD/DATA/light.20200720.b20.16_128/03.perambulator/", ".peram", [128, 128, 4, 4, 70, 70], 70
)

perambulator_charm_pre = preset.PerambulatorBinary(
    "/dg_hpc/LQCD/DATA/light.20200720.b20.16_128/03.perambulator.charm/", ".charm.peram", [128, 128, 4, 4, 70, 70], 70
)

# cfg = "s1.0_cfg_2000.stout.n20.f0.12.nev70"
dispatcher = Dispatch("cfglist.700.txt", "balabala")
for cfg in dispatcher:
    # You might need this
    # savepath = F"./DATA/corr_isoscalar_{cfg}.npy"
    # if os.path.exists(savepath):
    #     print(F"Jump: {cfg} exists!")
    #     continue
    elemental = elemental_pre.load(cfg)
    perambulator_light = perambulator_light_pre.load(cfg)
    perambulator_charm = perambulator_charm_pre.load(cfg)

    # D_D = QuarkDiagram([[0, 1], [2, 0]])
    # Ds_Ds = QuarkDiagram([[0, 2], [1, 0]])
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

    line_light = Propagator(perambulator_light, 128)
    line_charm = Propagator(perambulator_charm, 128)
    line_local_light = PropagatorLocal(perambulator_light, 128)
    D_src = Meson(elemental, op_D, True)
    D_snk = Meson(elemental, op_D, False)
    Ds_src = Meson(elemental, op_Ds, True)
    Ds_snk = Meson(elemental, op_Ds, False)

    import numpy as npo
    np = get_backend()
    t_snk = npo.arange(128)

    twopt = np.zeros((6, 128), "<c16")
    for t_src in range(128):
        print(t_src)
        tmp = compute_diagrams_multitime(
            [D_D, Ds_Ds, DDsbar_DDsbar_direct, DDsbar_DDsbar_cross, chic1_DDs, chic1_ccbar_eta],
            [t_src, t_src, t_snk, t_src,],
            [D_src, Ds_src, D_snk, Ds_snk],
            [None, line_charm, line_light, line_local_light],
        )
        twopt[0:6] += np.roll(tmp, -t_src, 1)[0:6]

    twopt /= 128
    print(twopt)
    print(np.arccosh((np.roll(twopt, -1, 1) + np.roll(twopt, 1, 1)) / twopt / 2))