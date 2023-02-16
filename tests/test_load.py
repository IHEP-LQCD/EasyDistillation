import numpy as np
import os
import sys
test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))  # yapf: disable
from lattice import GaugeFieldIldg  # yapf: disable
from lattice import setBackend  # yapf: disable
# import cupy as cp


setBackend(np)

latt_size = [4, 4, 4, 8]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt
Nd = 4

gaugeDataNpy = np.load(os.path.join(
    test_dir, F"./weak_field.npy")).reshape(Nd, 8, 4, 4, 4, 3, 3)
gaugeDataBinary = np.fromfile(os.path.join(test_dir, F"./weak_field.bin"),
                              "<c16", count=np.prod([8, 4, 4, 4, Nd, 3, 3])).reshape(Nd, 8, 4, 4, 4, 3, 3)

# gauge ILDG file shape [Lt, Lz, Ly, Lx, Nd, Nc, Nc]
gaugeIldg = GaugeFieldIldg(os.path.join(
    test_dir, F"./weak_field"), ".lime", shape=[8, 4, 4, 4, Nd, 3, 3])

gaugeDataIldg = gaugeIldg.load(
    "")[1:3, :, 5:8, :, :, :, :].transpose(4, 0, 1, 2, 3, 5, 6)
print("gauge reading res 1 = ", np.linalg.norm(
    gaugeDataIldg - gaugeDataNpy[:, 1:3, :, 5:8, :, :, :]))

gaugeDataIldg = gaugeIldg.load("")[()].transpose(4, 0, 1, 2, 3, 5, 6)
print("gauge reading res 2 = ", np.linalg.norm(gaugeDataIldg - gaugeDataNpy))
print("gauge reading res 3 = ", np.linalg.norm(gaugeDataIldg - gaugeDataBinary))
