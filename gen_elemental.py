import cupy
from lattice import Dispatch, setBackend
import lattice
import dif_dict
import mom_dict
from time import time


setBackend(cupy)

lattSize = [16, 16, 16, 128]

confs = lattice.GaugeFieldTimeSlice("/dg_hpc/LQCD/DATA/light.20200720.b20.16_128/01.stout_smear/")
eigs = lattice.EigenVectorTimeSlice("/dg_hpc/LQCD/DATA/light.20200720.b20.16_128/02.laplace_eigs/")
difList = dif_dict.dictToList()
momList = mom_dict.dictToList()
outPrefix = r"/dg_hpc/LQCD/DATA/light.20200720.b20.16_128/04.meson.mom2=3/"
outSuffix = r".stout.n20.f0.12.nev70.meson"

elementals = lattice.ElementalGenerator(lattSize, confs, eigs, difList, momList)

res = cupy.zeros((128, len(difList), len(momList), 70, 70), "<c16")

dispatcher = Dispatch("cfglist.sorted.txt")

for cfg in dispatcher:
    if not cfg.endswith("00"):
        continue
    elem = elementals[cfg]
    print(cfg, end=" ")

    s = time()
    for t in range(128):
        res[t] = elem(t)
    print(f"{time()-s:.2f}Sec", end=" ")
    print(f"{elem.sizeInByte/elem.timeInSec/1024**2:.2f}MB/s")
    cupy.save(f"{outPrefix}{cfg}{outSuffix}.npy", res)
