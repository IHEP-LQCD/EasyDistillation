from lattice import set_backend, get_backend

set_backend("cupy")
backend = get_backend()
import lattice
from lattice.elemental import ElementalGenerator
from time import perf_counter

latt_size = [16, 16, 16, 128]
Lx, Ly, Lz, Lt = latt_size

confs = lattice.GaugeFieldIldg(
    R"/dg_hpc/LQCD/DATA/clqcd_nf2_clov_L16_T128_b2.0_ml-0.05862_sn2_srho0.12_gg5.65_gf5.2_usg0.780268_usf0.949104/00.cfgs/clqcd_nf2_clov_L16_T128_b2.0_xi5_ml-0.05862_cfg_",
    ".lime", [128, 16**3, 4, 3, 3]
)
eigs = lattice.EigenVectorNpy(
    R"/dg_hpc/LQCD/shichunjiang/DATA/clqcd_nf2_clov_L16_T128_b2.0_ml-0.05862_sn2_srho0.12_gg5.65_gf5.2_usg0.780268_usf0.949104/02.laplace_eigs/clqcd_nf2_clov_L16_T128_b2.0_xi5_ml-0.05862_cfg_",
    ".lime.npy", [128, 70, 16**3, 3], 70
)
# eigs = lattice.EigenVectorNpy(R"./aaa.", ".evecs.npy", [70, 128, 16**3 * 3], 70)
mom_list = lattice.mom_dict.mom_dict_to_list(9)
out_prefix = R"./aaa."
out_suffix = R".elemental.npy"

elementals = ElementalGenerator(latt_size, confs, eigs, 1, mom_list)

import numpy

data = numpy.zeros((Lt, elementals.num_derivative, elementals.num_momentum, elementals.Ne, elementals.Ne), "<c16")
for cfg in ["2000"]:
    print(cfg, end=" ")
    s = perf_counter()
    elementals.load(cfg)
    for t in range(Lt):
        data[t] = elementals.calc(t).get()

    print(f"{perf_counter() - s:.2f}Sec", end=" ")
    numpy.save(f"{out_prefix}{cfg}{out_suffix}", data.transpose(1, 2, 0, 3, 4))
