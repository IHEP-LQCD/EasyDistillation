import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))


from lattice import set_backend, get_backend, check_QUDA

grid_size = [1, 1, 2, 2]
# check and init PyQuda & MPI
if not check_QUDA(grid_size=grid_size, backend="cupy", resource_path=None): 
    raise ImportError("Please install PyQuda")
latt_size = [4, 4, 4, 8]
Lx, Ly, Lz, Lt = latt_size
Gx, Gy, Gz, Gt = grid_size
Lx, Ly, Lz, Lt = [Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt]
Ne = 20
Ns = 4


from lattice import PerambulatorGenerator, PerambulatorNpy
from pyquda import enum_quda, getMPIRank
from pyquda.core import gatherLattice

set_backend("cupy")
backend = get_backend()

from lattice import GaugeFieldIldg, EigenvectorNpy, Nc, Nd


gauge_field = GaugeFieldIldg(f"{test_dir}/", ".lime", [Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Nd, Nc, Nc])
eigenvector = EigenvectorNpy(f"{test_dir}/", ".eigenvector.input.npy", [Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Nc], Ne)

perambulator = PerambulatorGenerator(
    latt_size=latt_size,
    gauge_field=gauge_field,
    eigenvector=eigenvector,
    mass=0.09253,
    tol=1e-9,
    maxiter=1000,
    xi_0=4.8965,
    nu=0.86679,
    clover_coeff_t=0.8549165664,
    clover_coeff_r=2.32582045,
    t_boundary=-1,  # for this test lattice, use t_boundary=-1
    multigrid=False,
    contract_prec="<c16"
)  # arbitrary dirac parameters
perambulator.dirac.invert_param.verbosity = enum_quda.QudaVerbosity.QUDA_SUMMARIZE

out_prefix = "tests/"
out_suffix = ".perambulator.npy"


def check(cfg, data):
    data_ref = PerambulatorNpy(out_prefix, out_suffix, [Lt * Gt, Lt * Gt, Ns, Ns, Ne, Ne], Ne).load(cfg)
    res = backend.linalg.norm(data_ref[:] - backend.array(data))
    print(f"Test cfg {cfg}, res = {res}")

# save all timeslices in one
import numpy
peramb = numpy.zeros((Lt * Gt, Lt, Ns, Ns, Ne, Ne), "<c16")
for cfg in ["weak_field"]:
    print(cfg)
    perambulator.load(cfg)
    perambulator.stout_smear(20, 0.1)
    for t in range(Lt * Gt):
        peramb[t] = perambulator.calc(t).get()
    # mpi gather lattice data and save
    # Note: For perambulator, mpi gather always gather timeslices and reduce space!
    peramb_h = gatherLattice(peramb, axes = [1, -1, -1, -1], reduce_op="sum", root=0)
    if getMPIRank() == 0:
        for t in range(Lt * Gt):
            peramb_h[t] = numpy.roll(peramb_h[t], -t, 0)
        # numpy.save(f"{out_prefix}{cfg}{out_suffix}", peramb_h)

        # check data
        check(cfg, peramb_h)

# save timeslices seprately
# import numpy
# peramb = numpy.zeros((1, Lt, Ns, Ns, Ne, Ne), "<c16")
# for cfg in ["weak_field"]:
#     print(cfg)
#     perambulator.load(cfg)
#     perambulator.stout_smear(20, 0.1)
#     for t in range(Lt * Gt):
#         peramb[0] = perambulator.calc(t).get()
#         # mpi gather lattice data and save
#         # Note: For perambulator, mpi gather always gather timeslices and reduce space!
#         peramb_h = gatherLattice(peramb[0], axes = [0, -1, -1, -1], reduce_op="sum", root=0)
#         if getMPIRank() == 0:
#             peramb_h[0] = numpy.roll(peramb_h[0], -t, 0)
#             numpy.save(f"{out_prefix}{cfg}.t{t:03d}.{out_suffix}", peramb_h[0])

perambulator.dirac.destroy()
