import re
from time import time
from typing import Tuple

from .abstract import FileMetaData, FileData, File
from ..backend import get_backend, get_numpy


def prod(a):
    p = 1
    for i in a:
        p *= i
    return p


class BinaryFileData(FileData):
    def __init__(self, file: str, elem: FileMetaData) -> None:
        self.file = file
        self.shape = elem.shape
        self.dtype = elem.dtype
        self.stride = [prod(self.shape[i:]) for i in range(1, len(self.shape))] + [1]
        self.bytes = int(re.match(r"^[<>=]?[iufc](?P<bytes>\d+)$", elem.dtype).group("bytes"))
        self.time_in_sec = 0.0
        self.size_in_byte = 0

    def get_count(self, key: Tuple[int]):
        return self.stride[len(key) - 1]

    def get_offset(self, key: Tuple[int]):
        offset = 0
        for a, b in zip(key, self.stride[0:len(key)]):
            offset += a * b
        return offset * self.bytes

    def __getitem__(self, key: Tuple[int]):
        numpy = get_backend()
        numpy_ori = get_numpy()
        s = time()
        # ret = numpy.asarray(
        #     loader(
        #         self.file,
        #         dtype=self.dtype,
        #         shape=tuple(self.shape),
        #         offset=0,
        #     )[key]
        # )  # yapf: disable
        ret = numpy.asarray(
            numpy_ori.memmap(
                self.file,
                dtype=self.dtype,
                mode="r",
                offset=0,
                shape=tuple(self.shape),
            )[key].copy()
        )
        self.time_in_sec += time() - s
        self.size_in_byte += ret.nbytes
        return ret


class BinaryFile(File):
    def __init__(self) -> None:
        self.file: str = None
        self.data: BinaryFileData = None

    def get_file_data(self, name: str, elem: FileMetaData) -> BinaryFileData:
        if self.file != name:
            self.file = name
            self.data = BinaryFileData(name, elem)
        return self.data
