from copy import deepcopy
from typing import List

from .filedata.abstract import FileData, FileMetaData
from .filedata.binary import BinaryFile
from .filedata.ildg import IldgFile
from .filedata.timeslice import QDPLazyDiskMapObjFile
from .filedata.ndarray import NdarrayFile, NdarrayTimeslicesFile


class GaugeField:
    def __init__(self, elem: FileMetaData) -> None:
        self.elem = deepcopy(elem)


class Eigenvector:
    def __init__(self, elem: FileMetaData, eigenNum: int) -> None:
        self.elem = deepcopy(elem)
        self.Ne = eigenNum


class Elemental:
    def __init__(self, elem: FileMetaData, eigenNum: int) -> None:
        self.elem = deepcopy(elem)
        self.Ne = eigenNum


class Perambulator:
    def __init__(self, elem: FileMetaData, eigenNum: int) -> None:
        self.elem = deepcopy(elem)
        self.Ne = eigenNum


class OnePoint:
    def __init__(self, elem: FileMetaData) -> None:
        self.elem = deepcopy(elem)


class TwoPoint:
    def __init__(self, elem: FileMetaData) -> None:
        self.elem = deepcopy(elem)


class GaugeFieldTimeSlice(QDPLazyDiskMapObjFile, GaugeField):
    def __init__(self, prefix: str, suffix: str, shape: List[int] = [128, 4, 16**3, 3, 3]) -> None:
        super().__init__()
        GaugeField.__init__(self, FileMetaData(shape, ">c16", 2))
        self.prefix = prefix
        self.suffix = ".stout.n20.f0.12.mod" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class EigenvectorTimeSlice(QDPLazyDiskMapObjFile, Eigenvector):
    def __init__(self, prefix: str, suffix: str, shape: List[int] = [128, 70, 16**3, 3], totNe: int = 70) -> None:
        super().__init__()
        Eigenvector.__init__(self, FileMetaData(shape, ">c8", 2), totNe)
        self.prefix = prefix
        self.suffix = ".stout.n20.f0.12.laplace_eigs.3d.mod" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class EigenvectorNpy(NdarrayFile, Eigenvector):
    def __init__(self, prefix: str, suffix: str, shape: List[int] = [70, 128, 16**3, 3], totNe: int = 70) -> None:
        super().__init__()
        Eigenvector.__init__(self, FileMetaData(shape, "<c16", 2), totNe)
        self.prefix = prefix
        self.suffix = ".lime.npy" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class EigenvectorHostmem(NdarrayFile, Eigenvector):
    def __init__(self, host_ndarray, shape: List[int] = [70, 128, 16**3, 3], totNe: int = 70) -> None:
        super().__init__()
        if host_ndarray.shape != shape:
            raise ValueError(
                f"Please check that host_ndarray shape {host_ndarray.shape} does not match expected shape (totNe, Lt, Lz * Ly * Lx, Nc) = {shape}"
            )
        self.Ne = totNe
        # Initialize data using reference, NOT memcopy of data.
        self.data = host_ndarray

    def load(self, key: str = None):
        return self.data


class PerambulatorBinary(BinaryFile, Perambulator):
    def __init__(self, prefix: str, suffix: str, shape: List[int] = [128, 128, 4, 4, 70, 70], totNe: int = 70) -> None:
        super().__init__()
        Perambulator.__init__(self, FileMetaData(shape, "<c16", 0), totNe)
        self.prefix = prefix
        self.suffix = ".stout.n20.f0.12.nev70.peram" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class PerambulatorNpy(NdarrayFile, Perambulator):
    def __init__(self, prefix: str, suffix: str, shape: List[int] = [128, 128, 4, 4, 70, 70], totNe: int = 70) -> None:
        super().__init__()
        Perambulator.__init__(self, FileMetaData(shape, "<c8", 0), totNe)
        self.prefix = prefix
        self.suffix = ".stout.n20.f0.12.nev70.peram" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class PerambulatorTimeslicesNpy(NdarrayTimeslicesFile, Perambulator):
    """
    this Perambulator data class is modified for timeslide solely saved  data.
    """

    def __init__(self, prefix: str, suffix: str, shape: List[int] = [128, 128, 4, 4, 70, 70], totNe: int = 70) -> None:
        super().__init__()
        Perambulator.__init__(self, FileMetaData(shape, "<c8", 0), totNe)
        self.prefix = prefix
        self.suffix = ".stout.n20.f0.12.nev70.peram" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class ElementalBinary(BinaryFile, Elemental):
    def __init__(self, prefix: str, suffix: str, shape: List[int] = [40, 27, 128, 70, 70], totNe: int = 70) -> None:
        super().__init__()
        Elemental.__init__(self, FileMetaData(shape, "<c16", 0), totNe)
        self.prefix = prefix
        self.suffix = ".stout.n20.f0.12.nev70.meson" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class Jpsi2gammaBinary(BinaryFile, TwoPoint):
    def __init__(self, prefix: str, suffix: str, shape: List[int] = [128, 2, 3, 4, 27, 128]) -> None:
        super().__init__()
        TwoPoint.__init__(self, FileMetaData(shape, "<f8", 0))
        self.prefix = prefix
        self.suffix = ".mesonspec.2pt.bin" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class GaugeFieldIldg(IldgFile, GaugeField):
    def __init__(self, prefix: str, suffix: str, shape: List[int] = [128, 16**3, 4, 3, 3]) -> None:
        super().__init__()
        GaugeField.__init__(self, FileMetaData(shape, ">c16", 0))
        self.prefix = prefix
        self.suffix = ".lime" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class GaugeFieldBinary(BinaryFile, GaugeField):
    def __init__(self, prefix: str, suffix: str, shape: List[int] = [128, 16**3, 4, 3, 3], dtype: str = "<f8") -> None:
        super().__init__()
        GaugeField.__init__(self, FileMetaData(shape, dtype, 0))
        self.prefix = prefix
        self.suffix = ".dat" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class ElementalNpy(NdarrayFile, Elemental):
    def __init__(self, prefix: str, suffix: str, shape: List[int] = [4, 123, 128, 70, 70], totNe: int = 70) -> None:
        super().__init__()
        Elemental.__init__(self, FileMetaData(shape, "<c8", 0), totNe)
        self.prefix = prefix
        self.suffix = ".stout.n20.f0.12.nev70.meson.npy" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class Jpsi2gammaNpy(NdarrayFile, TwoPoint):
    def __init__(self, prefix: str, suffix: str) -> None:
        super().__init__()
        TwoPoint.__init__(self, None)
        self.prefix = prefix
        self.suffix = ".2pt.npy" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class OnePointNpy(NdarrayFile, OnePoint):
    def __init__(self, prefix: str, suffix: str) -> None:
        super().__init__()
        # [2, 123, 128]
        OnePoint.__init__(self, None)
        self.prefix = prefix
        self.suffix = ".1pt.npy" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)
