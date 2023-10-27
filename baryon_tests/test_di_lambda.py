#!/public/home/xinghy/anaconda3-2023.03/bin/python
import os
import sys
import numpy as np
import time

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, "/public/home/gengyq/EasyDistillation"))

from lattice import set_backend, get_backend

set_backend("cupy")

Lt = 64
Lx = 32
conf_id = 58150
###############################################################################
from lattice.insertion.mom_dict import momDict_mom1
from lattice.insertion import (
    Insertion,
    Operator,
    GammaName,
    DerivativeName,
    ProjectionName,
)

ins_P = Insertion(GammaName.B1, DerivativeName.IDEN, ProjectionName.T1, momDict_mom1)
op_P = Operator("proton", [ins_P[1](0, 0, 0)], [1])
print("ins_P", ins_P.rows)
###############################################################################
from lattice import preset

elemental = preset.ElementalNpy(
    f"/public/home/gengyq/Proton_v2/data/{conf_id}/C32P29.",
    ".VVV.npy",
    [64, 100, 100, 100],
    100,
)
perambulator_u = preset.PerambulatorNpy(
    f"/public/home/gengyq/Proton_v2/data/{conf_id}/C32P29.",
    ".peramb.light.npy",
    [64, 64, 4, 4, 100, 100],
    100,
)
perambulator_s = preset.PerambulatorNpy(
    f"/public/home/gengyq/Proton_v2/data/{conf_id}/C32P29.",
    ".peramb.strange.npy",
    [64, 64, 4, 4, 100, 100],
    100,
)
###############################################################################
from lattice.quark_diagram import (
    BaryonDiagram,
    compute_diagrams_multitime,
    Baryon,
    Propagator,
)

P_src = Baryon(elemental, op_P, True)
P_snk = Baryon(elemental, op_P, False)

peramb_u = Propagator(perambulator_u, Lt)
peramb_s = Propagator(perambulator_s, Lt)
###############################################################################

P_src.load(conf_id)
P_snk.load(conf_id)
peramb_u.load(conf_id)
peramb_s.load(conf_id)

P_P1 = BaryonDiagram(
    [[0, 0, 0, 0], [[1, 2, 3], 0, 0, 0], [0, 0, 0, 0], [0, 0, [1, 2, 3], 0]],
    None,
)

P_P2 = BaryonDiagram(
    [[0, 0, 0, 0], [[1, 2, 3], 0, 0, 0], [0, 0, 0, 0], [0, 0, [1, 2, 3], 0]],
    [1, 4, 2, 2, 3, 3],
)

P_P3 = BaryonDiagram(
    [[0, 0, 0, 0], [[1, 2, 3], 0, 0, 0], [0, 0, 0, 0], [0, 0, [1, 2, 3], 0]],
    [1, 1, 2, 2, 3, 6],
)

# P_P4 = BaryonDiagram(
#     [[0, 0, 0, 0], [[1, 2, 3], 0, 0, 0], [0, 0, 0, 0], [0, 0, [1, 2, 3], 0]],
#     [1, 4, 2, 2, 3, 6],
# )

# P_P5 = BaryonDiagram(
#     [[0, 0, 0, 0], [[1, 2, 3], 0, 0, 0], [0, 0, 0, 0], [0, 0, [1, 2, 3], 0]],
#     [1, 1, 2, 5, 3, 3],
# )

# P_P6 = BaryonDiagram(
#     [[0, 0, 0, 0], [[1, 2, 3], 0, 0, 0], [0, 0, 0, 0], [0, 0, [1, 2, 3], 0]],
#     [1, 4, 2, 5, 3, 3],
# )

# P_P7 = BaryonDiagram(
#     [[0, 0, 0, 0], [[1, 2, 3], 0, 0, 0], [0, 0, 0, 0], [0, 0, [1, 2, 3], 0]],
#     [1, 1, 2, 5, 3, 6],
# )

# P_P8 = BaryonDiagram(
#     [[0, 0, 0, 0], [[1, 2, 3], 0, 0, 0], [0, 0, 0, 0], [0, 0, [1, 2, 3], 0]],
#     [1, 4, 2, 5, 3, 6],
# )

print("diagram set done")

###############################################################################
# t_snk = np.arange(Lt)

backend = get_backend()
twopt = backend.zeros((3, Lt, 4, 4, 4, 4), "<c16")

for t_src in range(1):
    st1 = time.time()
    # peram_all_light[t_src] = np.roll(peram_all_light[t_src], t_src, 0)
    for t_snk in range(Lt):
        tmp = compute_diagrams_multitime(
            [P_P1, P_P2, P_P3],
            [t_snk, t_src, t_snk, t_src],
            [P_snk, P_src, P_snk, P_src],
            [None, peramb_u, peramb_u, peramb_s],
            "",
        )
        # print(tmp[0], tmp[1], tmp[2])
        if t_snk < t_src:
            tmp = -tmp
        twopt[:, (t_snk - t_src) % Lt] += tmp

        # print(tmp[0], tmp[1], tmp[2])

        # del peram_u
        # cp._default_memory_pool.free_all_blocks()

    # if one want to cauculate per source to all sink, please use below:

    # tmp = backend.roll(tmp, -t_src, 1)
    # tmp[0, Lt - t_src : Lt] = -tmp[0, Lt - t_src : Lt]
    # twopt += tmp

np.save("/public/home/gengyq/laph/Lambda/result/di_lambda_diagram_dirac.npy", twopt)
di_baryon = np.zeros((Lt, 4, 4, 4, 4), "<c16")
di_baryon = -2 * twopt[0] + 2 * twopt[2] + 4 * twopt[1]
print(di_baryon)
np.save("/public/home/gengyq/laph/Lambda/result/di_lambda_dirac.npy", di_baryon)
#     ed1 = time.time()
#     print(f"time{t_src} caululate done, time used: %.3f s" % (ed1 - st1))
# print(twopt)
# np.save("/public/home/gengyq/Proton_v/result/proton.npy", twopt)
