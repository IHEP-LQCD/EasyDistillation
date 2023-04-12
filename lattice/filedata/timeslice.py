from io import BufferedReader
import re
import struct
from time import time
from typing import Dict, Tuple
import xml.etree.ElementTree as ET

from .abstract import FileMetaData, FileData, File
from ..backend import get_backend


def read_str(f: BufferedReader) -> str:
    length = struct.unpack(">i", f.read(4))[0]
    return f.read(length).decode("utf-8")


def read_tuple(f: BufferedReader) -> Tuple[int]:
    length = struct.unpack(">i", f.read(4))[0]
    cnt = length // 4
    fmt = ">" + "i" * cnt
    return struct.unpack(fmt, f.read(4 * cnt))


def read_pos(f: BufferedReader) -> int:
    return struct.unpack(">qq", f.read(16))[1]


def prod(a):
    p = 1
    for i in a:
        p *= i
    return p


class QDPLazyDiskMapObjFileData(FileData):
    def __init__(self, file: str, elem: FileMetaData, offsets: Dict[Tuple[int], int], xml_tree: ET.ElementTree) -> None:
        self.file = file
        self.shape = elem.shape[elem.extra:]
        self.dtype = elem.dtype
        self.extra = elem.extra
        self.extraShape = elem.shape[0:elem.extra]
        self.offsets = offsets
        latt_size = [int(x) for x in xml_tree.find("lattSize").text.split(" ")]
        self.latt_size = latt_size.copy()
        decay_dir = int(xml_tree.find("decay_dir").text)
        assert decay_dir == 3
        self.stride = [prod(self.shape[i:]) for i in range(1, len(self.shape))] + [1]
        self.bytes = int(re.match(r"^[<>=]?[iufc](?P<bytes>\d+)$", elem.dtype).group("bytes"))
        self.time_in_sec = 0.0
        self.size_in_byte = 0

    def get_count(self, key: Tuple[int]):
        if key == ():
            return prod(self.shape)
        else:
            return self.stride[len(key) - 1]

    def get_offset(self, key: Tuple[int]):
        offset = 0
        for a, b in zip(key, self.stride[0:len(key)]):
            offset += a * b
        return offset * self.bytes

    def __getitem__(self, key: Tuple[int]):
        import numpy as numpy_ori
        numpy = get_backend()
        if isinstance(key, int):
            key = (key, )
        if key[0:self.extra] not in self.offsets:
            raise IndexError(f"index {key} is out of bounds for axes")
        else:
            s = time()
            ret = numpy.asarray(
                numpy_ori.memmap(
                    self.file,
                    dtype=self.dtype,
                    mode="r",
                    offset=self.offsets[key[:self.extra]],
                    shape=tuple(self.shape),
                )[key[self.extra:]].copy().astype("<c8")
            )  # yapf: disable
            self.time_in_sec += time() - s
            self.size_in_byte += ret.nbytes
            return ret


class QDPLazyDiskMapObjFile(File):
    def __init__(self) -> None:
        self.magic: str = "XXXXQDPLazyDiskMapObjFileXXXX"
        self.file: str = None
        self.version: int = 0
        self.mod_meta_data: ET.ElementTree = None
        self.data: QDPLazyDiskMapObjFileData = None

    def read_meta_data(self, f: BufferedReader) -> Dict[Tuple[int], int]:
        assert self.magic == read_str(f)
        self.version = struct.unpack(">i", f.read(4))[0]
        xml_tree = ET.ElementTree(ET.fromstring(read_str(f)))
        f.seek(read_pos(f))
        num_record = struct.unpack(">I", f.read(4))[0]
        offsets: Dict[Tuple[int], int] = {}
        for _ in range(num_record):
            key = read_tuple(f)
            val = read_pos(f)
            offsets[key] = val
        return offsets, xml_tree

    def get_file_data(self, name: str, elem: FileMetaData) -> QDPLazyDiskMapObjFileData:
        if self.file != name:
            self.file = name
            with open(name, "rb") as f:
                offsets, xml_tree = self.read_meta_data(f)
            self.data = QDPLazyDiskMapObjFileData(name, elem, offsets, xml_tree)
        return self.data
