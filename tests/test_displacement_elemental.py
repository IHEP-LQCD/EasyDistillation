import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
print(test_dir)
sys.path.insert(0, os.path.join(test_dir, ".."))
from time import perf_counter
from lattice import set_backend, get_backend

set_backend("numpy")
backend = get_backend()

from lattice import GaugeFieldIldg, EigenvectorNpy, ElementalNpy, DisplacementElementalGenerator, Nd, Nc

latt_size = [4, 4, 4, 8]
Lx, Ly, Lz, Lt = latt_size
Ne = 20
distance = 8

gauge_field = GaugeFieldIldg(F"{test_dir}/", R".lime", [Lt, Lz, Ly, Lx, Nd, Nc, Nc])
eigenvector = EigenvectorNpy(F"{test_dir}/", R".eigenvector.npy", [Lt, Ne, Lz, Ly, Lx, Nc], Ne)

mom_list = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1), (0, 1, 2), (1, 1, 2)]
num_mom = len(mom_list)
elemental = DisplacementElementalGenerator(latt_size, gauge_field, eigenvector, distance, mom_list)
out_prefix = F"{test_dir}/"
out_suffix = ".displacement_elemental.npy"


def check(cfg, data):
    data_ref = ElementalNpy(out_prefix, out_suffix, [distance + 1, num_mom, Lt, Ne, Ne], Ne).load(cfg)[:]
    res = backend.linalg.norm(data_ref - data)
    print(F"Test cfg {cfg}, res = {res}")


data = backend.zeros((Lt, distance + 1, num_mom, Ne, Ne), "<c16")
for cfg in ["weak_field"]:
    print(cfg)

    elemental.load(cfg)
    for t in range(Lt):
        s = perf_counter()
        data[t] = elemental.calc(t)
        print(f"EASYDISTILLATION: {perf_counter() - s:.2f}sec to calculate elemental at t={t}")

    # backend.save(F"{out_prefix}{cfg}{out_suffix}", data.transpose(1, 2, 0, 3, 4))
    check(cfg, data.transpose(1, 2, 0, 3, 4))
