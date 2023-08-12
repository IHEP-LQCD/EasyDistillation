import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))
from lattice import set_backend, get_backend, Nd, Nc
from lattice import GaugeFieldIldg

set_backend("numpy")
backend = get_backend()

latt_size = [4, 4, 4, 8]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt

# numpy load Ndarray file
gaugeDataNpy = backend.load(os.path.join(test_dir, "weak_field.npy")).reshape(
    Nd, Lt, Lz, Ly, Lx, Nc, Nc
)

# numpy load binary file
gaugeDataBinary = backend.fromfile(
    os.path.join(test_dir, "weak_field.bin"),
    "<c16",
    count=backend.prod([Nd, Lt, Lz, Ly, Lx, Nc, Nc]),
).reshape(Nd, Lt, Lz, Ly, Lx, Nc, Nc)

# !!! Note: gauge ILDG file shape [Lt, Lz, Ly, Lx, Nd, Nc, Nc] <- chroma convention
gaugeIldg = GaugeFieldIldg(
    os.path.join(test_dir, "weak_field"), ".lime", shape=[Lt, Lz, Ly, Lx, Nd, Nc, Nc]
)

gaugeDataIldg = gaugeIldg.load("")[1:3, :, 5:8, :, :, :, :].transpose(
    4, 0, 1, 2, 3, 5, 6
)

# check load results.
check1 = backend.linalg.norm(gaugeDataIldg - gaugeDataNpy[:, 1:3, :, 5:8, :, :, :])
print("gauge reading res 1 = ", check1)

gaugeDataIldg = gaugeIldg.load("")[:].transpose(4, 0, 1, 2, 3, 5, 6)
check2 = backend.linalg.norm(gaugeDataIldg - gaugeDataNpy)
print("gauge reading res 2 = ", check2)

check3 = backend.linalg.norm(gaugeDataIldg - gaugeDataBinary)
print("gauge reading res 3 = ", check3)

if check1 == check2 == check3 == 0:
    print("Test ends, pass!")
else:
    print("Error: Test not pass!")
