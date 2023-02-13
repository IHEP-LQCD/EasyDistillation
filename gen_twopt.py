import cupy as np
from opt_einsum import contract
import lattice
from time import time

lattice.setBackend(np)

perambulator = lattice.PerambulatorBinary(
    R"/hpcfs/lqcd/qcd/rqzhang/new_charm/perambulator/peram-",
    ".lime",
)
elemental = lattice.ElementalNpy(
    R"/hpcfs/lqcd/qcd/xyjiang/elementals/DATA/charm.b28.16_128.wo_stout.corrected/04.meson/",
    R".npy",
)

gamma0 = lattice.gamma(0)
gamma1 = lattice.gamma(1)
gamma4 = lattice.gamma(8)
gamma5 = lattice.gamma(15)
Gamma = gamma5

dispatcher = lattice.Dispatch("cfglist.only.txt")
for cfg in dispatcher:
    p = perambulator.load(cfg)  # (128, 128, 4, 4, 50, 50)
    e = elemental.load(cfg)  # (9, 27, 128, 50, 50)

    s = time()
    twopt = np.zeros((9, 27, 128), "<c16")
    elem = e[:]
    for t in lattice.processBar(range(128)):
        pera = p[t]
        for d in range(9):
            twopt[d] += contract(
                "tjiba,jk,ptbc,tklcd,li,pad->pt",
                pera.conj(),
                gamma5 @ Gamma,
                np.roll(elem[d], -t, 1),
                pera,
                gamma4 @ Gamma.T.conj() @ gamma4 @ gamma5,
                elem[d, :, t].conj(),
            )
    twopt /= 128
    print(f"Total: {time() - s:.2f}sec;", end=" ")
    print(f"Perambulator: {p.timeInSec:.2f}sec with {p.sizeInByte / 1024**2 / p.timeInSec:.2f}MB/s;", end=" ")
    print(f"Elemental: {e.timeInSec:.2f}sec with {e.sizeInByte / 1024**2 / e.timeInSec:.2f}MB/s;", end=" ")
    print("")

import matplotlib.pyplot as plt

twopt = twopt.real
for d in range(9):
    plt.plot(np.arange(127).get(), np.log(twopt[d, 0, :-1] / twopt[d, 0, 1:]).get(), ',-', label=f"d={d}")
plt.ylim(0.2, 0.4)
plt.xlim(0, 50)
plt.legend()
plt.savefig("gen_twopt.svg")
