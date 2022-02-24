from math import comb
from typing import List, Tuple

from .abstract import FileData
from .backend import getBackend
from .timeslice import GaugeFieldTimeSlice, EigenVecsTimeSlice


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
    U = None
    V = None
    VPV = None
    difList = None
    momList = None
    # cacheD = None
    einsum = None

    @staticmethod
    def prepare(difList: List[Tuple[int]], momList: List[Tuple[int]]):
        from opt_einsum import contract

        numpy = getBackend()
        ElementalUtil.U = numpy.zeros((4, 16 ** 3, 3, 3), "<c16")
        ElementalUtil.V = numpy.zeros((70, 16 ** 3, 3), "<c8")
        ElementalUtil.VPV = numpy.zeros((len(difList), len(momList), 70, 70), "<c16")
        ElementalUtil.difList = difList
        ElementalUtil.momList = momList
        ElementalUtil.einsum = contract

    # @staticmethod
    # def clearCache():
    #     ElementalUtil.cacheD = {}

    @staticmethod
    def D(V, U, d):
        numpy = getBackend()
        Vf = numpy.roll(V.reshape(70, 16, 16, 16, 3), -1, 3 - d).reshape(70, -1, 3)
        UVf = ElementalUtil.einsum("xab,exb->exa", U[d], Vf)
        UVb = ElementalUtil.einsum("xba,exb->exa", U[d].conj(), V)
        UVb = numpy.roll(UVb.reshape(70, 16, 16, 16, 3), 1, 3 - d).reshape(70, -1, 3)
        return UVf - UVb

    # @staticmethod
    # def nD(V, U, ds):
    #     if ds in ElementalUtil.cacheD:
    #         ret = ElementalUtil.cacheD[ds]
    #     else:
    #         print(ds, end=" ")
    #         ret = V
    #         for d in ds:
    #             ret = ElementalUtil.D(ret, U, d)
    #         ElementalUtil.cacheD[ds] = ret
    #     return ret

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
        # ElementalUtil.clearCache()
        if ElementalUtil.difList != [()]:
            for d in range(U.shape[0]):
                U[d] = self.U[t, d]
        for e in range(V.shape[0]):
            V[e] = self.V[t, e]
        for idif, dif in enumerate(ElementalUtil.difList):
            VPV[idif] = 0
            for lr in range(len(dif) + 1):
                coeff = (-1) ** lr * comb(len(dif), lr)
                right = ElementalUtil.nD(V, U, dif[:lr])
                left = ElementalUtil.nD(V, U, dif[lr:][::-1])
                for imom, mom in enumerate(ElementalUtil.momList):
                    VPV[idif, imom] += ElementalUtil.einsum("x,exc,fxc->ef", coeff * self.P[mom], left.conj(), right)
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
        gaugeField: GaugeFieldTimeSlice,
        eigenVecs: EigenVecsTimeSlice,
        difList: List[Tuple[int]] = [()],
        momList: List[Tuple[int]] = [(0, 0, 0)],
    ) -> None:
        self.lattSize = lattSize
        self.U = gaugeField
        self.V = eigenVecs
        self.P = MomentaPhase(*lattSize[0:3])
        ElementalUtil.prepare(difList, momList)

    def __getitem__(self, val: str):
        Udata = self.U[val]
        Vdata = self.V[val]
        assert self.lattSize == Udata.lattSize and self.lattSize == Vdata.lattSize
        return ElementalData(Udata, Vdata, self.P)
