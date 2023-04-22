from typing import List, Tuple

from lattice.filedata.abstract import FileData
from lattice.backend import get_backend
from lattice.preset import GaugeField, EigenVector

Nd = 4
Nc = 3


def prod(a):
    p = 1
    for i in a:
        p *= i
    return p


class MomentaPhase:
    def __init__(self, Lx: int, Ly: int, Lz: int) -> None:
        numpy = get_backend()
        self.X = numpy.arange(Lx).reshape(1, 1, Lx).repeat(Lz, 0).repeat(Ly, 1) * 2j * numpy.pi / Lx
        self.Y = numpy.arange(Ly).reshape(1, Ly, 1).repeat(Lz, 0).repeat(Lx, 2) * 2j * numpy.pi / Ly
        self.Z = numpy.arange(Lz).reshape(Lz, 1, 1).repeat(Ly, 1).repeat(Lx, 2) * 2j * numpy.pi / Lz
        self.cache = {}

    def calc(self, key: Tuple[int]):
        Px, Py, Pz = key
        if key not in self.cache:
            numpy = get_backend()
            self.cache[key] = numpy.exp(Px * self.X + Py * self.Y + Pz * self.Z)
        return self.cache[key]


class ElementalUtil:
    latt_size = None
    Ne = None
    U = None
    V = None
    Vd = None
    VPV = None
    distance = None
    momenta = None
    einsum = None

    @staticmethod
    def prepare(distance: int, momenta: List[Tuple[int]], latt_size: List[int], Ne: int):
        from opt_einsum import contract

        numpy = get_backend()
        Nx = prod(latt_size[0:3])
        ElementalUtil.latt_size = latt_size
        ElementalUtil.Ne = Ne
        ElementalUtil.U = numpy.zeros((Nd, Nx, Nc, Nc), "<c16")
        ElementalUtil.V = numpy.zeros((Ne, Nx, Nc), "<c8")
        ElementalUtil.Vd = numpy.zeros((2 * (Nd - 1), Ne, Nx, Nc), "<c16")
        ElementalUtil.VPV = numpy.zeros((distance + 1, len(momenta), Ne, Ne), "<c16")
        ElementalUtil.distance = distance
        ElementalUtil.momenta = momenta
        ElementalUtil.einsum = contract

    @staticmethod
    def D(V, U, distance):
        if distance == 0:
            return V
        elif distance == 1:
            numpy = get_backend()
            Vd = ElementalUtil.Vd
            Ne = ElementalUtil.Ne
            Lz, Ly, Lx = ElementalUtil.latt_size[0:3][::-1]
            for d in range(Nd - 1):
                tmp = numpy.roll(V.reshape(Ne, Lz, Ly, Lx, Nc), -1, 3 - d).reshape(Ne, -1, Nc)
                Vd[d] = ElementalUtil.einsum("xab,exb->exa", U[d], tmp)
                tmp = ElementalUtil.einsum("xba,exb->exa", U[d].conj(), V)
                Vd[-d - 1] = numpy.roll(tmp.reshape(Ne, Lz, Ly, Lx, Nc), 1, 3 - d).reshape(Ne, -1, Nc)
            return Vd.mean(0)
        else:
            numpy = get_backend()
            Vd = ElementalUtil.Vd
            Ne = ElementalUtil.Ne
            Lz, Ly, Lx = ElementalUtil.latt_size[0:3][::-1]
            for d in range(Nd - 1):
                tmp = numpy.roll(Vd[d].reshape(Ne, Lz, Ly, Lx, Nc), -1, 3 - d).reshape(Ne, -1, Nc)
                Vd[d] = ElementalUtil.einsum("xab,exb->exa", U[d], tmp)
                tmp = ElementalUtil.einsum("xba,exb->exa", U[d].conj(), Vd[-d - 1])
                Vd[-d - 1] = numpy.roll(tmp.reshape(Ne, Lz, Ly, Lx, Nc), 1, 3 - d).reshape(Ne, -1, Nc)
            return Vd.mean(0)


class ElementalData:
    def __init__(self, U: FileData, V: FileData, P: MomentaPhase) -> None:
        self.U = U
        self.V = V
        self.P = P

    def calc(self, t: int):
        U = ElementalUtil.U
        V = ElementalUtil.V
        VPV = ElementalUtil.VPV
        if ElementalUtil.distance > 0:
            for d in range(U.shape[0]):
                U[d] = self.U[t, :, d]
        for e in range(V.shape[0]):
            V[e] = self.V[t, e]
        for dist in range(ElementalUtil.distance + 1):
            VPV[dist] = 0
            right = ElementalUtil.D(V, U, dist)
            left = V
            for imom, mom in enumerate(ElementalUtil.momenta):
                VPV[dist, imom] += ElementalUtil.einsum(
                    "x,exc,fxc->ef",
                    self.P.calc(mom).reshape(-1),
                    left.conj(),
                    right,
                )
        return VPV

    @property
    def timeInSec(self):
        return self.U.time_in_sec + self.V.time_in_sec

    @property
    def sizeInByte(self):
        return self.U.size_in_byte + self.V.size_in_byte


class ElementalGenerator:
    def __init__(
        self,
        lattSize: List[int],
        gaugeField: GaugeField,
        eigenVecs: EigenVector,
        distance: int = 0,
        momenta: List[Tuple[int]] = [(0, 0, 0)],
    ) -> None:
        self.lattSize = lattSize
        self.U = gaugeField
        self.V = eigenVecs
        self.P = MomentaPhase(*lattSize[0:3])
        ElementalUtil.prepare(distance, momenta, lattSize, eigenVecs.Ne)

    def load(self, val: str):
        Udata = self.U.load(val)
        Vdata = self.V.load(val)
        assert self.lattSize == Udata.lattSize and self.lattSize == Vdata.lattSize
        return ElementalData(Udata, Vdata, self.P)


import lattice
from time import time

lattice.set_backend("cupy")
cupy = get_backend()

latt_size = [16, 16, 16, 128]

confs = lattice.GaugeFieldIldg(
    R"/hpcfs/lqcd/qcd/gongming/productions/charm.b28.16_128.wo_stout.corrected/",
    ".lime",
)
eigs = lattice.EigenVectorTimeSlice(
    R"/hpcfs/lqcd/qcd/rqzhang/new_charm/laplacevector_ihep/test.3d.eigs.mod-",
    ".lime",
)
distance = 8
mom_list = lattice.mom_dict.mom_dict_to_list(9)
outPrefix = R"DATA/charm.b28.16_128.wo_stout.corrected/04.meson/"
outSuffix = R".npy"

elementals = ElementalGenerator(latt_size, confs, eigs, distance, mom_list)

res = cupy.zeros((128, distance + 1, len(mom_list), 50, 50), "<c16")

dispatcher = lattice.Dispatch("cfglist.only.txt")

for cfg in dispatcher:
    elem = elementals.load(cfg)
    print(cfg, end=" ")

    s = time()
    for t in range(128):
        res[t] = elem.calc(t)
    print(f"{time() - s:.2f}Sec", end=" ")
    print(f"{elem.sizeInByte / elem.timeInSec / 1024 ** 2:.2f}MB/s")
    cupy.save(f"{outPrefix}{cfg}{outSuffix}", res.transpose(1, 2, 0, 3, 4))
