#!/public/home/xinghy/anaconda3-2023.03/bin/python

import numpy as np
import time

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
perambulator = preset.PerambulatorNpy(
    f"/public/home/gengyq/Proton_v2/data/{conf_id}/C32P29.",
    ".peramb.light.npy",
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

peramb_u = Propagator(perambulator, Lt)

###############################################################################

P_src.load(conf_id)
P_snk.load(conf_id)
peramb_u.load(conf_id)


P_P1 = BaryonDiagram(
    [
        [0, 0],
        [[1, 2, 3], 0],
    ],
    None,
)

P_P2 = BaryonDiagram(
    [
        [0, 0],
        [[1, 2, 3], 0],
    ],
    [1, 3],
)

print("diagram set done")
###############################################################################
# if one want to cauculate per source to all sink, please use t_snk below and delete loop of t_snk
# t_snk = np.arange(Lt)

backend = get_backend()
twopt = np.zeros((1, Lt), "<c16")
for t_src in range(Lt):
    st1 = time.time()
    # peram_all_light[t_src] = np.roll(peram_all_light[t_src], t_src, 0)
    for t_snk in range(Lt):
        tmp = compute_diagrams_multitime(
            [P_P1],
            [t_snk, t_src],
            [P_snk, P_src],
            [None, peramb_u, peramb_u, peramb_u],
            "pp",
        ) - compute_diagrams_multitime(
            [P_P2],
            [t_snk, t_src],
            [P_snk, P_src],
            [None, peramb_u, peramb_u, peramb_u],
            "pp",
        )

        if t_snk < t_src:
            tmp = -tmp
        twopt[0, (t_snk - t_src) % Lt] += tmp

    ed1 = time.time()
    print(f"time{t_src} caululate done, time used: %.3f s" % (ed1 - st1))
print(twopt)
np.save(f"/public/home/gengyq/Proton_v2/result/proton_{conf_id}.npy", twopt)
