import cupy
# cupy.cuda.Device(3).use()
from lattice import setBackend
setBackend(cupy)
import lattice
from lattice.elemental import ElementalGenerator
from time import perf_counter

lattSize = [16, 16, 16, 128]

confs = lattice.GaugeFieldIldg(
    R"/dg_hpc/LQCD/DATA/clqcd_nf2_clov_L16_T128_b2.0_ml-0.05862_sn2_srho0.12_gg5.65_gf5.2_usg0.780268_usf0.949104/00.cfgs/clqcd_nf2_clov_L16_T128_b2.0_xi5_ml-0.05862_cfg_",
    ".lime", [128, 16**3, 4, 3, 3]
)
eigs = lattice.EigenVectorNpy(
    R"/dg_hpc/LQCD/shichunjiang/DATA/clqcd_nf2_clov_L16_T128_b2.0_ml-0.05862_sn2_srho0.12_gg5.65_gf5.2_usg0.780268_usf0.949104/02.laplace_eigs/clqcd_nf2_clov_L16_T128_b2.0_xi5_ml-0.05862_cfg_",
    ".lime.npy", [70, 128, 16**3 * 3], 70
)
# eigs = lattice.EigenVectorNpy(R"./aaa.", ".evecs.npy", [70, 128, 16**3 * 3], 70)
momList = lattice.mom_dict.momDictToList(9)
outPrefix = R"./aaa."
outSuffix = R".elemental.npy"

elementals = ElementalGenerator(lattSize, confs, eigs, 1, momList)

for cfg in ["2000"]:
    print(cfg, end=" ")
    s = perf_counter()
    elementals.load(cfg)

    print(f"{perf_counter() - s:.2f}Sec", end=" ")
    cupy.save(f"{outPrefix}{cfg}{outSuffix}", elementals.elemental.transpose(1, 2, 0, 3, 4))
