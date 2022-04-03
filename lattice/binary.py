import re
from time import time
from typing import Tuple

from .abstract import FileMetaData, FileData, File
from .backend import getBackend, getNumpy
from .sliceloader import binloader as loader


def prod(a):
    p = 1
    for i in a:
        p *= i
    return p


class BinaryFileData(FileData):
    def __init__(self, file: str, elem: FileMetaData) -> None:
        self.file = file
        self.shape = elem.shape
        self.stride = [prod(self.shape[i:]) for i in range(1, len(self.shape))] + [1]
        self.dtype = elem.dtype
        self.bytes = int(re.match(r"^[<>=]?[iufc](?P<bytes>\d+)$", elem.dtype).group("bytes"))
        self.timeInSec = 0.0
        self.sizeInByte = 0

    def getCount(self, key: Tuple[int]):
        return self.stride[len(key) - 1]

    def getOffset(self, key: Tuple[int]):
        offset = 0
        for a, b in zip(key, self.stride[0:len(key)]):
            offset += a * b
        return offset * self.bytes

    def __getitem__(self, key: Tuple[int]):
        numpy = getBackend()
        s = time()
        ret = numpy.asarray(
            loader(
                self.file,
                dtype=self.dtype,
                shape=tuple(self.shape),
                offset=0,
            )[key]
        )  # yapf: disable
        self.timeInSec += time() - s
        self.sizeInByte += ret.nbytes
        return ret


class BinaryFile(File):
    def __init__(self) -> None:
        self.file: str = None
        self.data: BinaryFileData = None

    def getFileData(self, name: str, elem: FileMetaData) -> BinaryFileData:
        if self.file != name:
            self.file = name
            self.data = BinaryFileData(name, elem)
        return self.data
