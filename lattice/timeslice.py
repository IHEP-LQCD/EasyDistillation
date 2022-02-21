from io import BufferedReader
from math import prod
import re
import struct
from time import time
from typing import Dict, Tuple
import xml.etree.ElementTree as ET

from .abstract import ElementMetaData, FileData, File
from .backend import getBackend


def readStr(f: BufferedReader) -> str:
    length = struct.unpack(">i", f.read(4))[0]
    return f.read(length).decode("utf-8")


def readTuple(f: BufferedReader) -> Tuple[int]:
    length = struct.unpack(">i", f.read(4))[0]
    cnt = length // 4
    fmt = ">" + "i" * cnt
    return struct.unpack(fmt, f.read(4 * cnt))


def readPos(f: BufferedReader) -> int:
    return struct.unpack(">qq", f.read(16))[1]


class QDPLazyDiskMapObjFileData(FileData):
    def __init__(self, file: str, elem: ElementMetaData, offsets: Dict[Tuple[int], int], xmlTree: ET.ElementTree) -> None:
        self.file = file
        self.extra = elem.extra
        self.extraShape = elem.shape[0 : elem.extra]
        self.offsets = offsets
        lattSize = [int(x) for x in xmlTree.find("lattSize").text.split(" ")]
        self.lattSize = lattSize.copy()
        decay_dir = int(xmlTree.find("decay_dir").text)
        assert decay_dir == 3
        self.shape = elem.shape[elem.extra :]
        self.stride = [prod(self.shape[i:]) for i in range(1, len(self.shape))] + [1]
        self.dtype = elem.dtype
        self.bytes = int(re.match(r"^[<>=]?[iufc](?P<bytes>\d+)$", elem.dtype).group("bytes"))
        self.timeInSec = 0.0
        self.sizeInByte = 0

    def getCount(self, key: Tuple[int]):
        if key == ():
            return prod(self.shape)
        else:
            return self.stride[len(key) - 1]

    def getOffset(self, key: Tuple[int]):
        offset = 0
        for a, b in zip(key, self.stride[0 : len(key)]):
            offset += a * b
        return offset * self.bytes

    def __getitem__(self, key: Tuple[int]):
        numpy = getBackend()
        if isinstance(key, int):
            key = (key,)
        if key[0 : self.extra] not in self.offsets:
            raise IndexError(f"index {key} is out of bounds for axes")
        else:
            s = time()
            ret = numpy.fromfile(
                self.file,
                dtype=self.dtype,
                count=self.getCount(key[self.extra :]),
                offset=self.offsets[key[0 : self.extra]] + self.getOffset(key[self.extra :]),
            ).reshape(self.shape[len(key[self.extra :]) :])
            self.timeInSec += time() - s
            self.sizeInByte += ret.nbytes
            return ret


class QDPLazyDiskMapObjFile(File):
    def __init__(self) -> None:
        self.magic: str = "XXXXQDPLazyDiskMapObjFileXXXX"
        self.file: str = None
        self.version: int = 0
        self.modMetaData: ET.ElementTree = None
        self.data: QDPLazyDiskMapObjFileData = None

    def readMetaData(self, f: BufferedReader) -> Dict[Tuple[int], int]:
        assert self.magic == readStr(f)
        self.version = struct.unpack(">i", f.read(4))[0]
        xmlTree = ET.ElementTree(ET.fromstring(readStr(f)))
        f.seek(readPos(f))
        numRecord = struct.unpack(">I", f.read(4))[0]
        offsets: Dict[Tuple[int], int] = {}
        for _ in range(numRecord):
            key = readTuple(f)
            val = readPos(f)
            offsets[key] = val
        return offsets, xmlTree

    def getFileData(self, key: str, elem: ElementMetaData) -> QDPLazyDiskMapObjFileData:
        if self.file != key:
            self.file = key
            with open(key, "rb") as f:
                offsets, xmlTree = self.readMetaData(f)
            self.data = QDPLazyDiskMapObjFileData(key, elem, offsets, xmlTree)
        return self.data


class GaugeFieldTimeSlice(QDPLazyDiskMapObjFile):
    def __init__(self, directory: str) -> None:
        super().__init__()
        self.id = "gaugeFieldTimeSlice"
        self.prefix = f"{directory}/"
        self.suffix = ".stout.n20.f0.12.mod"

    def __getitem__(self, key: str):
        elem = ElementMetaData([128, 4, 16 ** 3, 3, 3], ">c16", 2)
        return super().getFileData(f"{self.prefix}{key}{self.suffix}", elem)


class EigenVecsTimeSlice(QDPLazyDiskMapObjFile):
    def __init__(self, directory: str) -> None:
        super().__init__()
        self.id = "eigenVecsTimeSlice"
        self.prefix = f"{directory}/"
        self.suffix = ".stout.n20.f0.12.laplace_eigs.3d.mod"

    def __getitem__(self, key: str):
        elem = ElementMetaData([128, 70, 16 ** 3, 3], ">c8", 2)
        return super().getFileData(f"{self.prefix}{key}{self.suffix}", elem)
