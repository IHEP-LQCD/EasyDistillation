from lattice import check_QUDA, set_backend, get_backend

set_backend("cupy")
cupy = get_backend()

if not check_QUDA():
    raise ImportError("No QUDA avaliable")

from lattice import GaugeFieldIldg, Nc
from lattice.laplace_eigs import EigenVectorGenerator

latt_size = [16, 16, 16, 128]
Lx, Ly, Lz, Lt = latt_size

gauge_field = GaugeFieldIldg(
    "/dg_hpc/LQCD/DATA/clqcd_nf2_clov_L16_T128_b2.0_ml-0.05862_sn2_srho0.12_gg5.65_gf5.2_usg0.780268_usf0.949104/00.cfgs/clqcd_nf2_clov_L16_T128_b2.0_xi5_ml-0.05862_cfg_",
    ".lime", [128, 16, 16, 16, 4, 3, 3]
)
eigen_vector = EigenVectorGenerator([16, 16, 16, 128], gauge_field, 70, 1e-7)

data = cupy.zeros((Lt, eigen_vector.Ne, Lz * Ly * Lx, Nc), "<c16")
for cfg in ["2000"]:
    print(cfg)

    out_prefix = R"./aaa."
    out_suffix = ".evecs.npy"
    eigen_vector.load(cfg)
    eigen_vector.stout_smear(10, 0.12)
    for t in range(128):
        data[t] = eigen_vector.calc(t)
    cupy.save(F"{out_prefix}{cfg}{out_suffix}", data)
