import cupy
# cupy.cuda.Device(3).use()
from lattice import set_backend

set_backend(cupy)
import lattice
from lattice.elemental import ElementalGenerator
from time import perf_counter

latt_size = [16, 16, 16, 128]

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

for cfg in ["2000"]:
    print(cfg, end=" ")
    s = perf_counter()
    elementals.load(cfg)

    print(f"{perf_counter() - s:.2f}Sec", end=" ")
    cupy.save(f"{out_prefix}{cfg}{out_suffix}", elementals.elemental.transpose(1, 2, 0, 3, 4))
