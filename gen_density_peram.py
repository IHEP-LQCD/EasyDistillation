from lattice import set_backend, get_backend, check_QUDA

if not check_QUDA():
    raise ImportError("Please install PyQuda")

from lattice import DensityPerambulatorGenerator
from pyquda import enum_quda

set_backend("cupy")
backend = get_backend()

from lattice import GaugeFieldIldg, EigenvectorTimeSlice, Nc, Nd

latt_size = [16, 16, 16, 128]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt
Ne = 70

gauge_field = GaugeFieldIldg(
    "/dg_hpc/LQCD/gongming/productions/confs/light.20200720.b20.16_128/", ".lime", [Lt, Lz, Ly, Lx, Nd, Nc, Nc]
)
eigenvector = EigenvectorTimeSlice(
    "/dg_hpc/LQCD/DATA/light.20200720.b20.16_128/02.laplace_eigs/",
    ".stout.n20.f0.12.laplace_eigs.3d.mod",
    [Lt, Ne, Lz * Ly * Lx, Nc],
    Ne,
)
perambulator = DensityPerambulatorGenerator(
    latt_size=latt_size,
    gauge_field=gauge_field,
    eigenvector=eigenvector,
    mass=0.09253,
    tol=1e-15,
    maxiter=1000,
    xi_0=4.8965,
    nu=0.86679,
    clover_coeff_t=0.8549165664,
    clover_coeff_r=2.32582045,
    t_boundary=-1,  # for this test lattice, use t_boundary=-1
    multigrid=[[4, 4, 4, 4], [4, 4, 4, 4]],
    gamma_list=[1, 2, 4, 8],
    momentum_list=[(0, 0, 0)],
)
perambulator.dirac.invert_param.verbosity = enum_quda.QudaVerbosity.QUDA_SUMMARIZE

out_prefix = "tests/"
out_suffix = ".perambulators.npy"

data = backend.zeros((Lt, Ne, Lz * Ly * Lx, Nc), "<c16")
for cfg in ["s1.0_cfg_2000"]:
    print(cfg)

    perambulator.load(cfg)
    peramb = perambulator.calc(0, 64, 32)

    print(peramb.shape)
    # backend.save(f"{out_prefix}{cfg}{out_suffix}", peramb)

perambulator.dirac.destroy()
