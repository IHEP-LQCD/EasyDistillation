from copy import deepcopy

from .abstract import ElementMetaData
from .binary import BinaryFile
from .ildg import IldgFile
from .timeslice import QDPLazyDiskMapObjFile
from .ndarray import NdarrayFile


class GaugeField:
    def __init__(self, elem: ElementMetaData) -> None:
        self.elem = deepcopy(elem)


class EigenVector:
    def __init__(self, elem: ElementMetaData, eigenNum: int) -> None:
        self.elem = deepcopy(elem)
        self.Ne = eigenNum


class Elemental:
    def __init__(self, elem: ElementMetaData, eigenNum: int) -> None:
        self.elem = deepcopy(elem)
        self.Ne = eigenNum


class Perambulator:
    def __init__(self, elem: ElementMetaData, eigenNum: int) -> None:
        self.elem = deepcopy(elem)
        self.Ne = eigenNum


class OnePoint:
    def __init__(self, elem: ElementMetaData) -> None:
        self.elem = deepcopy(elem)


class TwoPoint:
    def __init__(self, elem: ElementMetaData) -> None:
        self.elem = deepcopy(elem)


class GaugeFieldTimeSlice(QDPLazyDiskMapObjFile, GaugeField):
    def __init__(self, prefix: str, suffix: str) -> None:
        super().__init__()
        GaugeField.__init__(ElementMetaData([128, 4, 16**3, 3, 3], ">c16", 2))
        self.prefix = prefix
        self.suffix = ".stout.n20.f0.12.mod" if suffix is None else suffix

    def __getitem__(self, key: str):
        return super().getFileData(f"{self.prefix}{key}{self.suffix}", self.elem)


class EigenVectorTimeSlice(QDPLazyDiskMapObjFile, EigenVector):
    def __init__(self, prefix: str, suffix: str) -> None:
        super().__init__()
        EigenVector.__init__(ElementMetaData([128, 70, 16**3, 3], ">c8", 2), 70)
        self.prefix = prefix
        self.suffix = ".stout.n20.f0.12.laplace_eigs.3d.mod" if suffix is None else suffix

    def __getitem__(self, key: str):
        return super().getFileData(f"{self.prefix}{key}{self.suffix}", self.elem)


class PerambulatorBinary(BinaryFile, Perambulator):
    def __init__(self, prefix: str, suffix: str) -> None:
        super().__init__()
        Perambulator.__init__(ElementMetaData([128, 128, 4, 4, 70, 70], "<c16", 0), 70)
        self.prefix = prefix
        self.suffix = ".stout.n20.f0.12.nev70.peram" if suffix is None else suffix

    def __getitem__(self, key: str):
        return super().getFileData(f"{self.prefix}{key}{self.suffix}", self.elem)


class ElementalBinary(BinaryFile, Elemental):
    def __init__(self, prefix: str, suffix: str) -> None:
        super().__init__()
        Elemental.__init__(ElementMetaData([40, 27, 128, 70, 70], "<c16", 0), 70)
        self.prefix = prefix
        self.suffix = ".stout.n20.f0.12.nev70.meson" if suffix is None else suffix

    def __getitem__(self, key: str):
        return super().getFileData(f"{self.prefix}{key}{self.suffix}", self.elem)


class Jpsi2gammaBinary(BinaryFile, TwoPoint):
    def __init__(self, prefix: str, suffix: str) -> None:
        super().__init__()
        TwoPoint.__init__(ElementMetaData([128, 2, 3, 4, 27, 128], "<f8", 0))
        self.prefix = prefix
        self.suffix = ".mesonspec.2pt.bin" if suffix is None else suffix

    def __getitem__(self, key: str):
        return super().getFileData(f"{self.prefix}{key}{self.suffix}", self.elem)


class GaugeFieldIldg(IldgFile, GaugeField):
    def __init__(self, prefix: str, suffix: str) -> None:
        super().__init__()
        GaugeField.__init__(ElementMetaData([128, 16**3, 4, 3, 3], ">c16", 0))
        self.prefix = prefix
        self.suffix = ".lime" if suffix is None else suffix

    def __getitem__(self, key: str):
        return super().getFileData(f"{self.prefix}{key}{self.suffix}", self.elem)


class ElementalNpy(NdarrayFile, Elemental):
    def __init__(self, prefix: str, suffix: str) -> None:
        super().__init__()
        Elemental.__init__(None)
        self.prefix = prefix
        self.suffix = ".stout.n20.f0.12.nev70.meson.npy" if suffix is None else suffix

    def __getitem__(self, key: str):
        return super().getFileData(f"{self.prefix}{key}{self.suffix}", self.elem)


class Jpsi2gammaNpy(NdarrayFile, TwoPoint):
    def __init__(self, prefix: str, suffix: str) -> None:
        super().__init__()
        TwoPoint.__init__(None)
        self.prefix = prefix
        self.suffix = ".2pt.npy" if suffix is None else suffix

    def __getitem__(self, key: str):
        return super().getFileData(f"{self.prefix}{key}{self.suffix}", self.elem)


class OnePointNpy(NdarrayFile, OnePoint):
    def __init__(self, prefix: str, suffix: str) -> None:
        super().__init__()
        OnePoint.__init__(None)
        self.prefix = prefix
        self.suffix = ".1pt.npy" if suffix is None else suffix

    def __getitem__(self, key: str):
        # [2, 123, 128]
        return super().getFileData(f"{self.prefix}{key}{self.suffix}", self.elem)
