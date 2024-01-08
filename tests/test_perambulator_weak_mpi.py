import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
print(test_dir)
sys.path.insert(0, os.path.join(test_dir, ".."))

sys.path.insert(0, "/dg_hpc/LQCD/shichunjiang/PyQuda_devel")
os.environ["QUDA_RESOURCE_PATH"] = "/dg_hpc/LQCD/shichunjiang/PyQuda_devel/.cache"

from lattice import set_backend, get_backend, check_QUDA

grid_size = [1, 1, 1, 2]
if not check_QUDA(grid_size):
    raise ImportError("Please install PyQuda")
latt_size = [4, 4, 4, 8]
Lx, Ly, Lz, Lt = latt_size
Gx, Gy, Gz, Gt = grid_size
Lx, Ly, Lz, Lt = [Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt]
Ne = 20
Ns = 4

from pyquda import getGridCoord, getMPIRank

gx, gy, gz, gt = getGridCoord()

from lattice import PerambulatorGenerator, PerambulatorNpy
from pyquda import enum_quda, gather

set_backend("cupy")
backend = get_backend()

from lattice import GaugeFieldIldg, EigenvectorNpy, Nc, Nd


gauge_field = GaugeFieldIldg(f"{test_dir}/", ".lime", [Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Nd, Nc, Nc])
eigenvector = EigenvectorNpy(f"{test_dir}/", ".eigenvector.npy", [Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Nc], Ne)
perambulator = PerambulatorGenerator(
    latt_size, gauge_field, eigenvector, 0.09253, 1e-9, 1000, 4.8965, 0.86679, 0.8549165664, 2.32582045, True, False
)  # arbitrary dslash parameters
perambulator.dslash.invert_param.verbosity = enum_quda.QudaVerbosity.QUDA_SUMMARIZE

out_prefix = "tests/"
out_suffix = ".perambulator.npy"


def check(cfg, data):
    # data_ref = PerambulatorNpy(out_prefix, out_suffix, [Lt * Gt, Lt, Ns, Ns, Ne, Ne], Ne).load(cfg)[
    #     :, Lt * (gt) : Lt * (gt + 1), :, :, :, :
    # ]
    data_ref = PerambulatorNpy(out_prefix, out_suffix, [Lt * Gt, Lt * Gt, Ns, Ns, Ne, Ne], Ne).load(cfg)[
        :, :, :, :, :, :
    ]
    for t in range(Lt * Gt):
        data_ref[t] = backend.roll(data_ref[t], shift=+t, axis=0)
    res = backend.linalg.norm(data_ref[:, Lt * (gt) : Lt * (gt + 1)] - data)
    # res = backend.linalg.norm(data_ref - data)
    print(f"Test cfg {cfg}, res = {res}")


peramb = backend.zeros((Lt * Gt, Lt, Ns, Ns, Ne, Ne), "<c16")
for cfg in ["weak_field"]:
    print(cfg)
    perambulator.load(cfg)
    perambulator.stout_smear(20, 0.1)
    for t in range(Lt * Gt):
        peramb[t] = perambulator.calc(t)
    # import numpy
    # peramb = gather(peramb.get(), axes = [1, -1, -1, -1], root=0)
    # if getMPIRank == 0:
    #     for t in range(Lt * Gt):
    #         peramb[t] = numpy.roll(peramb[t], -t 0)
    #     numpy.save(f"{out_prefix}{cfg}{out_suffix}", peramb)
    check(cfg, peramb)

perambulator.dslash.destroy()
