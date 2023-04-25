from lattice.backend import set_backend, get_backend

set_backend("cupy")

from lattice import Dispatch, preset
from lattice.insertion.mom_dict import momDict_mom9
from lattice.insertion import Insertion, Operator, GammaName, DerivativeName, ProjectionName

from lattice import Meson, Propagator, PropagatorLocal, QuarkDiagram, compute_diagrams_multitime

from time import perf_counter
from lattice.correlator.two_particles import get_AB_opratorList_back2back, get_mom2_list
mom_max = 4   # compute mom2 = 0, 1, 2, 3

Nt = 128
backend = get_backend()

ins_D = Insertion(GammaName.PI, DerivativeName.IDEN, ProjectionName.A1, momDict_mom9)

ins_Dstar = Insertion(GammaName.RHO, DerivativeName.IDEN, ProjectionName.T1, momDict_mom9)

ins_chic1 = Insertion(GammaName.A1, DerivativeName.IDEN, ProjectionName.T1, momDict_mom9)

# # op_A = u_bar gamma5 c
# op_D = Operator("D", [ins_D[0](0, 0, 0)], [1])

# # op_B = c_bar gamma_i u
# op_Ds = Operator("Dbar_star", [ins_Dstar[2](0, 0, 0)], [1])

# # op_C = c_bar gamma_5 gamma_i c
# op_chic1 = Operator("chic1", [ins_chic1[2](0, 0, 0)], [1])


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
    [0, 0, 0, 0],
    [0, 0, 0, 3],
    [1, 0, 0, 0],
])

chic1_Jpsi_eta = QuarkDiagram([
    [0, 0, 1, 0],
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 3],
])

line_light = Propagator(perambulator_light, Nt)
line_charm = Propagator(perambulator_charm, Nt)
line_local_light = PropagatorLocal(perambulator_light, Nt)

# cfg = "s1.0_cfg_2000.stout.n20.f0.12.nev70"
dispatcher = Dispatch("cfglist.700.txt", "balabala")
for cfg in dispatcher:
    # You might need this to jump finished jobs.
    # savepath = F"./DATA/corr_isoscalar_{cfg}.npy"
    # if os.path.exists(savepath):
    #     print(F"Jump: {cfg} exists!")
    #     continue
    twopt_save = backend.zeros((mom_max, 6, Nt), "<c16")

    # You need use t_snk = numpy.arange(Nt) instead of backend.arange(Nt)
    import numpy
    t_snk = numpy.arange(Nt)

    for t_src in range(128):
        s = perf_counter()
        for i_mom2 in range(mom_max)[1:]:
            mom_list = get_mom2_list(i_mom2)
            op_D_list, op_Ds_list = get_AB_opratorList_back2back(ins_D[0], ins_Dstar[2], mom_list)

            for i_mom_pair_src in range(len(mom_list)):
                for i_mom_pair_snk in range(len(mom_list)):
                    D_src = Meson(elemental, op_D_list[i_mom_pair_src], True)
                    D_snk = Meson(elemental, op_D_list[i_mom_pair_snk], False)
                    Ds_src = Meson(elemental, op_Ds_list[i_mom_pair_src], True)
                    Ds_snk = Meson(elemental, op_Ds_list[i_mom_pair_snk], False)

                    line_light.load(cfg)
                    line_charm.load(cfg) 
                    line_local_light.load(cfg) 
                    D_src.load(cfg)
                    D_snk.load(cfg)
                    Ds_src.load(cfg)
                    Ds_snk.load(cfg)

                    tmp = compute_diagrams_multitime(
                        [D_D, Ds_Ds, DDsbar_DDsbar_direct, DDsbar_DDsbar_cross, chic1_DDs, chic1_Jpsi_eta],
                        [t_src, t_src, t_snk, t_snk],
                        [D_src, Ds_src, D_snk, Ds_snk],
                        [None, line_charm, line_light, line_local_light],
                    )
                    twopt_save[i_mom2, 0:6] += backend.roll(tmp, -t_src, 1)[0:6] 

            twopt_save[i_mom2] /= len(mom_list)**2
        print(t_src, F"{s - perf_counter(): .3f}")
    print(twopt_save)
    # print(backend.arccosh((backend.roll(twopt, -1, 1) + backend.roll(twopt, 1, 1)) / twopt / 2))
