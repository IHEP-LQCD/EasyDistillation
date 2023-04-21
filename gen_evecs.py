import cupy
from lattice import check_QUDA

if check_QUDA():
    from lattice.laplace_eigs import smear_gauge, calc_laplace_eigs
else:
    raise ImportError("No QUDA avaliable")

for cfg in ["2000"]:
    print(cfg)

    prefix = "/dg_hpc/LQCD/DATA/clqcd_nf2_clov_L16_T128_b2.0_ml-0.05862_sn2_srho0.12_gg5.65_gf5.2_usg0.780268_usf0.949104/00.cfgs/clqcd_nf2_clov_L16_T128_b2.0_xi5_ml-0.05862_cfg_"
    suffix = ".lime"
    out_prefix = R"./aaa."
    out_suffix = ".evecs.npy"
    gauge_spatial = smear_gauge(F"{prefix}{cfg}{suffix}", 10, 0.12)
    evecs = calc_laplace_eigs(gauge_spatial, 70, 1e-7)
    cupy.save(F"{out_prefix}{cfg}{out_suffix}", evecs)
