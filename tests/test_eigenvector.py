import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))
from time import perf_counter
from lattice import set_backend, get_backend

set_backend("cupy")
backend = get_backend()

from lattice import GaugeFieldIldg, EigenvectorNpy, EigenvectorGenerator, Nc, Nd

latt_size = [4, 4, 4, 8]
Lx, Ly, Lz, Lt = latt_size
Ne = 20

gauge_field = GaugeFieldIldg(f"{test_dir}/", ".lime", [Lt, Lz, Ly, Lx, Nd, Nc, Nc])

eigenvector = EigenvectorGenerator(latt_size, gauge_field, Ne, 1e-9)
out_prefix = f"{test_dir}/"
out_suffix = ".eigenvector.npy"


def check(cfg, evecs, evals):
    data_evecs_ref = EigenvectorNpy(out_prefix, out_suffix, [Lt, Ne, Lz, Ly, Lx, Nc], Ne).load(cfg)[:]
    data_evals_ref = backend.load(F"{out_prefix}{cfg}.eigenvalue.npy")
    res = 0
    for t in range(Lt):
        for e in range(Ne):
            phase = data_evecs_ref[t, e].reshape(-1)[0] / evecs[t, e].reshape(-1)[0]
            res += backend.linalg.norm(data_evecs_ref[t, e] / evecs[t, e] / phase - 1)
        print(f"Test cfg {cfg}, t = {t}, res = {res}")
    
    print(f"Test cfg {cfg}, eigen values, res = {backend.linalg.norm(data_evals_ref - evals)}")


eigne_vecs = backend.zeros((Lt, Ne, Lz, Ly, Lx, Nc), "<c16")
eigen_vals = backend.zeros((Lt, Ne), "<c16")
for cfg in ["weak_field"]:
    print(cfg)

    eigenvector.load(cfg)
    eigenvector.stout_smear(10, 0.12)
    # eigenvector.project_SU3()
    for t in range(Lt):
        s = perf_counter()
        eigne_vecs[t], eigen_vals[t] = eigenvector.calc(t)
        print(Rf"EASYDISTILLATION: {perf_counter()-s:.3f} sec to solve the lowest {Ne} eigensystem at t={t}.")

    # backend.save(F"{out_prefix}{cfg}.eigenvector.npy", eigne_vecs)
    # backend.save(F"{out_prefix}{cfg}.eigenvalue.npy", eigen_vals)
    check(cfg, eigne_vecs, eigen_vals)

print("Test ends!")
