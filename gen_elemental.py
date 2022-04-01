import numpy as cupy
import lattice
import dif_dict
import mom_dict
from time import time

lattice.setBackend(cupy)

lattSize = [16, 16, 16, 128]

confs = lattice.GaugeFieldTimeSlice(R"LQCD/DATA/light.20200720.b20.16_128/01.stout_smear/", None)
eigs = lattice.EigenVectorTimeSlice(R"LQCD/DATA/light.20200720.b20.16_128/02.laplace_eigs/", None)
difList = dif_dict.dictToList()[0:4]
momList = mom_dict.dictToList()[0:19]
outPrefix = R"LQCD/DATA/light.20200720.b20.16_128/04.meson.ndarray/"
outSuffix = R".stout.n20.f0.12.nev70.meson"

elementals = lattice.ElementalGenerator(lattSize, confs, eigs, difList, momList)

res = cupy.zeros((128, len(difList), len(momList), 70, 70), "<c16")

dispatcher = lattice.Dispatch("cfglist.sorted.txt")

for cfg in dispatcher:
    if not cfg.endswith("00"):
        continue
    elem = elementals[cfg]
    print(cfg, end=" ")

    s = time()
    for t in range(128):
        print(t)
        res[t] = elem(t)
    print(f"{time() - s:.2f}Sec", end=" ")
    print(f"{elem.sizeInByte / elem.timeInSec / 1024 ** 2:.2f}MB/s")
    cupy.save(f"{outPrefix}{cfg}{outSuffix}.npy", res.transpose(1, 2, 0, 3, 4))
