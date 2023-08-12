from io import SEEK_CUR, BufferedReader
import re
import struct
from time import time
from typing import Dict, Tuple
import xml.etree.ElementTree as ET

from .abstract import FileMetaData, FileData, File
from ..backend import get_backend


def prod(a):
    p = 1
    for i in a:
        p *= i
    return p


class IldgFileData(FileData):
    def __init__(self, file: str, elem: FileMetaData, offset: Tuple[int], xmlTree: ET.ElementTree) -> None:
        self.file = file
        self.shape = elem.shape
        self.dtype = elem.dtype
        self.offset = offset[0]
        tag = re.match(r"\{.*\}", xmlTree.getroot().tag).group(0)
        self.latt_size = [
            int(xmlTree.find(f"{tag}lx").text),
            int(xmlTree.find(f"{tag}ly").text),
            int(xmlTree.find(f"{tag}lz").text),
            int(xmlTree.find(f"{tag}lt").text),
        ]
        self.stride = [prod(self.shape[i:]) for i in range(1, len(self.shape))] + [1]
        self.bytes = int(re.match(r"^[<>=]?[iufc](?P<bytes>\d+)$", elem.dtype).group("bytes"))
        assert self.bytes == int(xmlTree.find(f"{tag}precision").text) // 8 * 2
        assert prod(elem.shape) * self.bytes == offset[1]
        self.timeInSec = 0.0
        self.sizeInByte = 0

    def get_count(self, key: Tuple[int]):
        return self.stride[len(key) - 1]

    def get_offset(self, key: Tuple[int]):
        offset = 0
        for a, b in zip(key, self.stride[0 : len(key)]):
            offset += a * b
        return offset * self.bytes

    def __getitem__(self, key: Tuple[int]):
        import numpy

        backend = get_backend()
        if isinstance(key, int):
            key = (key,)
        s = time()
        # fmt: off
        ret = backend.asarray(
            numpy.memmap(
                self.file,
                dtype=self.dtype,
                mode="r",
                offset=self.offset,
                shape=tuple(self.shape),
            )[key].copy().astype("<c16")
        )
        # fmt: on
        self.timeInSec += time() - s
        self.sizeInByte += ret.nbytes
        return ret


class IldgFile(File):
    def __init__(self) -> None:
        self.magic: str = b"\x45\x67\x89\xAB\x00\x01"
        self.file: str = None
        self.data: IldgFileData = None

    def read_meta_data(self, f: BufferedReader):
        obj_pos_size: Dict[str, Tuple[int]] = {}
        buffer = f.read(8)
        while buffer != b"":
            assert buffer.startswith(b"\x45\x67\x89\xAB\x00\x01")
            length = (struct.unpack(">Q", f.read(8))[0] + 7) // 8 * 8
            header = f.read(128).strip(b"\x00").decode("utf-8")
            obj_pos_size[header] = (f.tell(), length)
            f.seek(length, SEEK_CUR)
            buffer = f.read(8)

        offset = obj_pos_size["ildg-binary-data"]
        f.seek(obj_pos_size["ildg-format"][0])
        xml_tree = ET.ElementTree(ET.fromstring(f.read(obj_pos_size["ildg-format"][1]).strip(b"\x00").decode("utf-8")))
        return offset, xml_tree

    def get_file_data(self, name: str, elem: FileMetaData) -> IldgFileData:
        if self.file != name:
            self.file = name
            with open(name, "rb") as f:
                offset, xml_tree = self.read_meta_data(f)
            self.data = IldgFileData(name, elem, offset, xml_tree)
        return self.data
