import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
print(test_dir)
sys.path.insert(0, os.path.join(test_dir, ".."))

from lattice import set_backend, get_backend, check_QUDA

if not check_QUDA():
    raise ImportError("Please install PyQuda")

from lattice import PerambulatorGenerator, PerambulatorNpy
from pyquda import enum_quda

set_backend("cupy")
backend = get_backend()

from lattice import GaugeFieldIldg, EigenvectorNpy, Nc, Nd

latt_size = [4, 4, 4, 8]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt
Ne = 20
Ns = 4

gauge_field = GaugeFieldIldg(f"{test_dir}/", ".lime", [Lt, Lz, Ly, Lx, Nd, Nc, Nc])
eigenvector = EigenvectorNpy(f"{test_dir}/", ".eigenvector.npy", [Lt, Ne, Lz, Ly, Lx, Nc], Ne)
perambulator = PerambulatorGenerator(
    latt_size,
    gauge_field,
    eigenvector,
    0.09253,
    1e-9,
    1000,
    4.8965,
    0.86679,
    0.8549165664,
    2.32582045,
    True,
    False,
)  # arbitrary dslash parameters
perambulator.dslash.invert_param.verbosity = enum_quda.QudaVerbosity.QUDA_SUMMARIZE

out_prefix = "tests/"
out_suffix = ".perambulators.npy"


def check(cfg, data):
    data_ref = PerambulatorNpy(out_prefix, out_suffix, [Lt, Lt, Ns, Ns, Ne, Ne], Ne).load(cfg)[:]
    res = backend.linalg.norm(data_ref - data)
    print(f"Test cfg {cfg}, res = {res}")


peramb = backend.zeros((Lt, Lt, Ns, Ns, Ne, Ne), "<c16")
for cfg in ["weak_field"]:
    print(cfg)
    perambulator.load(cfg)
    for t in range(Lt):
        peramb[t] = perambulator.calc(t)
    # backend.save(f"{out_prefix}{cfg}{out_suffix}", peramb)
    check(cfg, peramb)

perambulator.dslash.destroy()
