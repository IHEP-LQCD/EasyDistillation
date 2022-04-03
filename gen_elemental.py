import cupy
import lattice
import dif_dict
import mom9_dict as mom_dict
from time import time

lattice.setBackend(cupy)

lattSize = [16, 16, 16, 128]

confs = lattice.GaugeFieldIldg(R"/hpcfs/lqcd/qcd/gongming/productions/charm.b28.16_128.wo_stout.corrected/", ".lime")
eigs = lattice.EigenVectorTimeSlice(R"/hpcfs/lqcd/qcd/rqzhang/new_charm/laplacevector_ihep/test.3d.eigs.mod-", ".lime")
difList = dif_dict.dictToList()[0:13]
momList = mom_dict.dictToList()
outPrefix = R"DATA/charm.b28.16_128.wo_stout.corrected/04.meson.deriv_2.mom2_max_9/"
outSuffix = R".npy"

elementals = lattice.ElementalGenerator(lattSize, confs, eigs, difList, momList)

res = cupy.zeros((128, len(difList), len(momList), 50, 50), "<c16")

dispatcher = lattice.Dispatch("cfglist.txt", "ehe")

for cfg in dispatcher:
    elem = elementals[cfg]
    print(cfg, end=" ")

    s = time()
    for t in range(128):
        res[t] = elem(t)
    print(f"{time() - s:.2f}Sec", end=" ")
    print(f"{elem.sizeInByte / elem.timeInSec / 1024 ** 2:.2f}MB/s")
    cupy.save(f"{outPrefix}{cfg}{outSuffix}", res.transpose(1, 2, 0, 3, 4))
