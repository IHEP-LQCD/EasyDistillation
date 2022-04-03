from math import factorial
from typing import List, Tuple

from .abstract import FileData
from .backend import getBackend
from .constant import Nd, Nc
from .preset import GaugeField, EigenVector


def prod(a):
    p = 1
    for i in a:
        p *= i
    return p


def comb(n, i):
    return factorial(n) // (factorial(i) * factorial(n - i))


class MomentaPhase:
    def __init__(self, Lx: int, Ly: int, Lz: int) -> None:
        numpy = getBackend()
        self.X = numpy.arange(Lx).reshape(1, 1, Lx).repeat(Lz, 0).repeat(Ly, 1) * 2j * numpy.pi / Lx
        self.Y = numpy.arange(Ly).reshape(1, Ly, 1).repeat(Lz, 0).repeat(Lx, 2) * 2j * numpy.pi / Ly
        self.Z = numpy.arange(Lz).reshape(Lz, 1, 1).repeat(Ly, 1).repeat(Lx, 2) * 2j * numpy.pi / Lz
        self.cache = {}

    def __call__(self, Px, Py, Pz):
        key = (Px, Py, Pz)
        if key not in self.cache:
            numpy = getBackend()
            self.cache[key] = numpy.exp(Px * self.X + Py * self.Y + Pz * self.Z)
        return self.cache[key]

    def __getitem__(self, key: Tuple[int]):
        return self.__call__(*key).reshape(-1)


class ElementalUtil:
    lattSize = None
    Nx = None
    Ne = None
    U = None
    V = None
    VPV = None
    derivs = None
    momenta = None
    einsum = None

    @staticmethod
    def prepare(derivs: List[Tuple[int]], momenta: List[Tuple[int]], lattSize: List[int], eigenNum: int):
        from opt_einsum import contract

        numpy = getBackend()
        Nx = prod(lattSize[0:3])
        Ne = eigenNum
        ElementalUtil.lattSize = lattSize
        ElementalUtil.Nx = Nx
        ElementalUtil.Ne = Ne
        ElementalUtil.U = numpy.zeros((Nd, Nx, Nc, Nc), "<c16")
        ElementalUtil.V = numpy.zeros((Ne, Nx, Nc), "<c8")
        ElementalUtil.VPV = numpy.zeros((len(derivs), len(momenta), Ne, Ne), "<c16")
        ElementalUtil.derivs = derivs
        ElementalUtil.momenta = momenta
        ElementalUtil.einsum = contract

    @staticmethod
    def D(V, U, d):
        numpy = getBackend()
        Ne = ElementalUtil.Ne
        Lz, Ly, Lx = ElementalUtil.lattSize[0:3][::-1]
        Vf = numpy.roll(V.reshape(Ne, Lz, Ly, Lx, Nc), -1, 3 - d).reshape(Ne, -1, Nc)
        UVf = ElementalUtil.einsum("xab,exb->exa", U[d], Vf)
        UVb = ElementalUtil.einsum("xba,exb->exa", U[d].conj(), V)
        UVb = numpy.roll(UVb.reshape(Ne, Lz, Ly, Lx, Nc), 1, 3 - d).reshape(Ne, -1, Nc)
        return UVf - UVb

    @staticmethod
    def nD(V, U, ds):
        ret = V
        for d in ds:
            ret = ElementalUtil.D(ret, U, d)
        return ret


class ElementalData:
    def __init__(self, U: FileData, V: FileData, P: MomentaPhase) -> None:
        self.U = U
        self.V = V
        self.P = P

    def __call__(self, t: int):
        U = ElementalUtil.U
        V = ElementalUtil.V
        VPV = ElementalUtil.VPV
        if ElementalUtil.derivs != [()]:
            for d in range(U.shape[0]):
                U[d] = self.U[t, :, d]
        for e in range(V.shape[0]):
            V[e] = self.V[t, e]
        for idrv, drv in enumerate(ElementalUtil.derivs):
            VPV[idrv] = 0
            for lr in range(len(drv) + 1):
                coeff = (-1)**lr * comb(len(drv), lr)
                right = ElementalUtil.nD(V, U, drv[:lr])
                left = ElementalUtil.nD(V, U, drv[lr:][::-1])
                for imom, mom in enumerate(ElementalUtil.momenta):
                    VPV[idrv, imom] += ElementalUtil.einsum("x,exc,fxc->ef", coeff * self.P[mom], left.conj(), right)
        # for idif, dif in enumerate(ElementalUtil.difList):
        #     VPV[idif] = 0
        #     for lr in range(len(dif) + 1):
        #         coeff = (-1) ** lr * comb(len(dif), lr)
        #         for imom, mom in enumerate(ElementalUtil.momList):
        #             right = ElementalUtil.einsum("x,exc->exc", coeff * self.P[mom], V)
        #             right = ElementalUtil.nD(right, U, dif[:lr])
        #             left = ElementalUtil.nD(V, U, dif[lr:][::-1])
        #             VPV[idif, imom] += ElementalUtil.einsum("exc,fxc->ef", left.conj(), right)
        return VPV

    @property
    def timeInSec(self):
        return self.U.timeInSec + self.V.timeInSec

    @property
    def sizeInByte(self):
        return self.U.sizeInByte + self.V.sizeInByte


class Elemental:
    def __init__(
        self,
        lattSize: List[int],
        gaugeField: GaugeField,
        eigenVecs: EigenVector,
        derivs: List[Tuple[int]] = [()],
        momenta: List[Tuple[int]] = [(0, 0, 0)],
    ) -> None:
        self.lattSize = lattSize
        self.U = gaugeField
        self.V = eigenVecs
        self.P = MomentaPhase(*lattSize[0:3])
        ElementalUtil.prepare(derivs, momenta, lattSize, eigenVecs.Ne)

    def __getitem__(self, val: str):
        Udata = self.U[val]
        Vdata = self.V[val]
        assert self.lattSize == Udata.lattSize and self.lattSize == Vdata.lattSize
        return ElementalData(Udata, Vdata, self.P)
