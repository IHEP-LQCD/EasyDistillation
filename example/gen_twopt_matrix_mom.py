# Example: h_c / eta_c GEVP
# compute a 2 by 2 twopt matrix with multi-mom in momlist.

from lattice import set_backend, get_backend
from lattice import preset
from lattice import Dispatch
from lattice.correlator.one_particle import twopoint_matrix_multi_mom
import os

set_backend("cupy")
backend = get_backend()

###############################################################################
from lattice.insertion.mom_dict import momDict_mom9
from lattice.insertion import Insertion, Operator, GammaName, DerivativeName, ProjectionName

ins1_hc = Insertion(GammaName.B1, DerivativeName.IDEN, ProjectionName.T1, momDict_mom9)
print("1+- oprator: ", ins1_hc[2])

ins1_jpsi = Insertion(GammaName.RHO, DerivativeName.IDEN, ProjectionName.T1, momDict_mom9)
print("1-- oprator: ", ins1_jpsi[2])

momlist = [
    (0, 0, -1),
    (0, -1, -1),
    (-1, -1, -1),
    (0, 0, -2),
    (0, -1, -2),
    (-1, -1, -2),
    (0, -2, -2),
]

elemental = preset.ElementalNpy(
    "/dg_hpc/LQCD/DATA/light.20200720.b20.16_128/04.meson.deriv1.mom9/",
    ".stout.n20.f0.12.nev70.meson.npy",
    [4, 123, 128, 70, 70],
    70,
)

perambulator_charm = preset.PerambulatorNpy(
    "/dg_hpc/LQCD/DATA/light.20200720.b20.16_128/03.perambulator.charm/",
    ".stout.n20.f0.12.nev70.charm.peram.npy",
    [128, 128, 4, 4, 70, 70],
    70,
)

dispatch = Dispatch("./cfglist.689.txt", "2pt")
save_dir = f"./"
os.system(f"mkdir -p {save_dir}")
for cfg in dispatch:
    save_path = f"{save_dir}{cfg}.2pt.npy"
    if os.path.exists(save_path):
        continue

    e = elemental.load(cfg)
    p = perambulator_charm.load(cfg)

    backend = get_backend()

    # compute 2pt
    twopt = twopoint_matrix_multi_mom([ins1_hc[2], ins1_jpsi[2]], momlist, e, p, list(range(128)), 128)  # [Nop, Lt]
    backend.save(save_path, twopt)
