from time import time
from typing import Tuple

from .abstract import ElementMetaData, FileData, File
from .backend import getBackend


class NdarrayFileData(FileData):
    def __init__(self, file: str, elem: ElementMetaData) -> None:
        self.file = file
        self.timeInSec = 0.0
        self.sizeInByte = 0

    def __getitem__(self, key: Tuple[int]):
        numpy = getBackend()
        s = time()
        ret = numpy.load(self.file, "r")[key]
        self.timeInSec += time() - s
        self.sizeInByte += ret.nbytes
        return ret


class NdarrayFile(File):
    def __init__(self) -> None:
        self.file: str = None
        self.data: NdarrayFileData = None

    def getFileData(self, key: str, elem: ElementMetaData) -> FileData:
        if self.file != key:
            self.file = key
            self.data = NdarrayFileData(key, elem)
        return self.data


class Elemental(NdarrayFile):
    def __init__(self, directory: str) -> None:
        super().__init__()
        self.prefix = f"{directory}/"
        self.suffix = ".stout.n20.f0.12.nev70.meson.npy"

    def __getitem__(self, key: str):
        return super().getFileData(f"{self.prefix}{key}{self.suffix}", None)


class Jpsi2gamma(NdarrayFile):
    def __init__(self, directory: str) -> None:
        super().__init__()
        self.prefix = f"{directory}/"
        self.suffix = ".2pt.npy"

    def __getitem__(self, key: str):
        return super().getFileData(f"{self.prefix}{key}{self.suffix}", None)


class OnePoint(NdarrayFile):
    def __init__(self, directory: str) -> None:
        super().__init__()
        self.prefix = f"{directory}/"
        self.suffix = ".1pt.npy"

    def __getitem__(self, key: str):
        # [2, 123, 128]
        return super().getFileData(f"{self.prefix}{key}{self.suffix}", None)
