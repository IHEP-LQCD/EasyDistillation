from lattice import Dispatch, QUDA
if QUDA():
    from lattice.laplace_eigs import laplace_eigs
else:
    raise ImportError("No QUDA avaliable")

key = 0
lightkey = [
    "-0.05766",
    "-0.05862",
    "-0.05945",
    "-0.06016",
][key]

if __name__ == "__main__":
    dispatcher = Dispatch("cfglist.txt", suffix=f"2.8-evecs-{key}")
    for cfg in dispatcher:
        print(cfg)

        prefix = F"/dg_hpc/LQCD/DATA/clqcd_nf2_clov_L16_T128_b2.0_ml{lightkey}_sn2_srho0.12_gg5.65_gf5.2_usg0.780268_usf0.949104/00.cfgs/clqcd_nf2_clov_L16_T128_b2.0_xi5_ml{lightkey}_cfg_"
        suffix = ".lime"
        out_prefix = F"clqcd_nf2_clov_L16_T128_b2.0_xi5_ml{lightkey}_cfg_"
        out_suffix = ".lime.npy"
        laplace_eigs(F"{prefix}{cfg}{suffix}", F"{out_prefix}{cfg}{out_suffix}", 10, 0.12, 10, 1e-7)
