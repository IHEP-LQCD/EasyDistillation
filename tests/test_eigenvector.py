import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))
from time import perf_counter
from lattice import set_backend, get_backend, check_QUDA

set_backend("numpy")
backend = get_backend()

# if not check_QUDA():
#     raise ImportError("No QUDA avaliable")

from lattice import GaugeFieldIldg, EigenvectorNpy, EigenvectorGenerator, Nc, Nd

latt_size = [4, 4, 4, 8]
Lx, Ly, Lz, Lt = latt_size
Ne = 20

gauge_field = GaugeFieldIldg(F"{test_dir}/", ".lime", [Lt, Lz, Ly, Lx, Nd, Nc, Nc])

eigenvector = EigenvectorGenerator(latt_size, gauge_field, Ne, 1e-9)
out_prefix = F"{test_dir}/"
out_suffix = ".evecs.npy"


def check(cfg, data):
    data_ref = EigenvectorNpy(out_prefix, out_suffix, [Lt, Ne, Lz * Ly * Lx, Nc], Ne).load(cfg)[:]
    res = 0
    for t in range(Lt):
        for e in range(Ne):
            phase = data_ref[t, e].reshape(-1)[0] / data[t, e].reshape(-1)[0]
            res += backend.linalg.norm(data_ref[t, e] / data[t, e] / phase - 1)
        print(res)


data = backend.zeros((Lt, Ne, Lz * Ly * Lx, Nc), "<c16")
for cfg in ["weak_field"]:
    print(cfg)

    eigenvector.load(cfg)
    eigenvector.stout_smear(10, 0.12)
    eigenvector.porject_SU3()
    print(backend.linalg.norm(eigenvector._U - backend.load("tests/weak_field.stout.npy")[:Nd - 1]))
    for t in range(Lt):
        s = perf_counter()
        data[t] = eigenvector.calc(t)
        print(FR"EASYDISTILLATION: {perf_counter()-s:.3f} sec to solve the lowest {Ne} eigensystem at t={t}.")

    # backend.save(F"{out_prefix}{cfg}{out_suffix}", data)
    check(cfg, data)
