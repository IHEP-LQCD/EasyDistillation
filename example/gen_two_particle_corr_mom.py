from lattice.backend import set_backend, get_backend

from time import perf_counter

set_backend("cupy")

from lattice import Dispatch, preset
from lattice.insertion.mom_dict import momDict_mom9
from lattice.insertion import Insertion, Operator, GammaName, DerivativeName, ProjectionName

from lattice import Meson, Propagator, PropagatorLocal, QuarkDiagram, compute_diagrams_multitime

from time import perf_counter
from lattice.correlator.two_particles import get_AB_opratorlist_row, get_mom2_list

mom_max = 4  # compute mom2 = 0, 1, 2, 3

Nt = 128
backend = get_backend()

ins_D = Insertion(GammaName.PI, DerivativeName.IDEN, ProjectionName.A1, momDict_mom9)

ins_Dstar = Insertion(GammaName.RHO, DerivativeName.IDEN, ProjectionName.T1, momDict_mom9)

ins_chic1 = Insertion(GammaName.A1, DerivativeName.IDEN, ProjectionName.T1, momDict_mom9)

# Read peramulators and elemental from file
elemental = preset.ElementalNpy(
    "/dg_hpc/LQCD/DATA/light.20200720.b20.16_128/04.meson.deriv1.mom9/",
    ".stout.n20.f0.12.nev70.meson.npy",
    [4, 123, 128, 70, 70],
    70,
)
perambulator_light = preset.PerambulatorBinary(
    "/dg_hpc/LQCD/DATA/light.20200720.b20.16_128/03.perambulator/",
    ".stout.n20.f0.12.nev70.peram",
    [128, 128, 4, 4, 70, 70],
    70,
)

perambulator_charm = preset.PerambulatorNpy(
    "/dg_hpc/LQCD/DATA/light.20200720.b20.16_128/03.perambulator.charm/",
    ".stout.n20.f0.12.nev70.charm.peram.npy",
    [128, 128, 4, 4, 70, 70],
    70,
)

# define diagrams: adjcency matrix
D_D = QuarkDiagram(
    [
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [2, 0, 0, 0],
        [0, 0, 0, 0],
    ]
)
Ds_Ds = QuarkDiagram(
    [
        [0, 0, 0, 0],
        [0, 0, 0, 2],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
    ]
)
DDsbar_DDsbar_direct = QuarkDiagram(
    [
        [0, 0, 1, 0],
        [0, 0, 0, 2],
        [2, 0, 0, 0],
        [0, 1, 0, 0],
    ]
)
DDsbar_DDsbar_cross = QuarkDiagram(
    [
        [0, 0, 2, 0],
        [3, 0, 0, 0],
        [0, 0, 0, 3],
        [0, 2, 0, 0],
    ]
)

chic1_DDs = QuarkDiagram(
    [
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 3],
        [1, 0, 0, 0],
    ]
)

chic1_Jpsi_eta = QuarkDiagram(
    [
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 3],
    ]
)

line_light = Propagator(perambulator_light, Nt)
line_charm = Propagator(perambulator_charm, Nt)
line_local_light = PropagatorLocal(perambulator_light, Nt)

dispatcher = Dispatch("cfglist.689.txt", "balabala")
for cfg in dispatcher:
    # You might need this to jump finished jobs.
    save_path = f"./data_2pt/2pt_{cfg}.npy"
    # if os.path.exists(savepath):
    #     print(F"Jump: {cfg} exists!")
    #     continue

    # calculation 2pt dimension: [hadron type * momentum list, diagrams]
    mom_list = []
    for i_mom2 in range(mom_max)[1:2]:
        mom_list += get_mom2_list(i_mom2)
    # hadron_type: e.g. this script use (A, B) = [ (ins_D[0], ins_Dstar[2]) ] as ONE hadron_type
    #              e.g. (A, B) = [ (ins_D[0], ins_Dstar[2]), (ins_Dstar[2], ins_Dstar[2]) ] ...
    op_D_list, op_Ds_list = get_AB_opratorlist_row(ins_D[0], ins_Dstar[2], mom_list)

    # e.g. this script calculate [src: hadron type * momentum list, sink: hadron type * momentum list, diagrams = 6]
    twopt_tosave = backend.zeros((1 * len(mom_list), 1 * len(mom_list), 6, Nt), "<c16")

    # You need use t_snk = numpy.arange(Nt) instead of backend.arange(Nt)
    import numpy

    t_snk = numpy.arange(Nt)

    ######################################################
    # make source and sink hadron instantiation list
    # !! this have to be done before calculation, in order to initialize only one cache pool !!
    # !! Otherwise Particle Class will redundantly reload cache!!
    D_src_list = [None] * len(mom_list)
    D_snk_list = [None] * len(mom_list)
    Ds_src_list = [None] * len(mom_list)
    Ds_snk_list = [None] * len(mom_list)
    for i_mom_pair in range(len(mom_list)):
        D_src_list[i_mom_pair] = Meson(elemental, op_D_list[i_mom_pair], True)
        D_snk_list[i_mom_pair] = Meson(elemental, op_D_list[i_mom_pair], False)
        Ds_src_list[i_mom_pair] = Meson(elemental, op_Ds_list[i_mom_pair], True)
        Ds_snk_list[i_mom_pair] = Meson(elemental, op_Ds_list[i_mom_pair], False)
    ######################################################

    ######################################################
    # calculate source timeslices loop
    for t_src in range(128):
        s = perf_counter()
        for i_mom_pair_src in range(len(mom_list)):
            for i_mom_pair_snk in range(len(mom_list)):
                D_src = D_src_list[i_mom_pair_src]
                D_snk = D_snk_list[i_mom_pair_snk]
                Ds_src = Ds_src_list[i_mom_pair_src]
                Ds_snk = Ds_snk_list[i_mom_pair_snk]

                line_light.load(cfg)
                line_charm.load(cfg)
                line_local_light.load(cfg)
                D_src.load(cfg)
                D_snk.load(cfg)
                Ds_src.load(cfg)
                Ds_snk.load(cfg)

                tmp = compute_diagrams_multitime(
                    [D_D, Ds_Ds, DDsbar_DDsbar_direct, DDsbar_DDsbar_cross, chic1_DDs, chic1_Jpsi_eta],  # 6 diagrams
                    [t_src, t_src, t_snk, t_snk],
                    [D_src, Ds_src, D_snk, Ds_snk],
                    [None, line_charm, line_light, line_local_light],
                )
                twopt_tosave[i_mom_pair_src, i_mom_pair_snk, 0:6] += backend.roll(tmp, -t_src, 1)[0:6]

        print("timer: ", t_src, f"{perf_counter() - s: .3f}")
    ######################################################

    print(twopt_tosave)
    # backend.save(save_path, twopt_tosave)
    # print(backend.arccosh((backend.roll(twopt, -1, 1) + backend.roll(twopt, 1, 1)) / twopt / 2))
