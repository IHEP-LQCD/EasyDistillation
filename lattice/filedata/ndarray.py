from time import time
from typing import Tuple

from .abstract import FileMetaData, FileData, File
from ..backend import get_backend


class NdarrayFileData(FileData):
    def __init__(self, file: str, elem: FileMetaData) -> None:
        self.file = file
        self.shape = elem.shape
        self.dtype = elem.dtype
        self.time_in_sec = 0.0
        self.size_in_byte = 0

    def __getitem__(self, key: Tuple[int]):
        import numpy

        backend = get_backend()
        s = time()
        # fmt: off
        # ret = numpy.asarray(
        #     loader(
        #         self.file,
        #     )[key]
        # )
        # fmt: on
        ret = backend.asarray(
            numpy.load(
                self.file,
                mmap_mode="r",
            )[key].copy()
        )
        self.time_in_sec += time() - s
        self.size_in_byte += ret.nbytes
        return ret


class NdarrayFile(File):
    def __init__(self) -> None:
        self.file: str = None
        self.data: NdarrayFileData = None

    def get_file_data(self, name: str, elem: FileMetaData) -> NdarrayFileData:
        if self.file != name:
            self.file = name
            self.data = NdarrayFileData(name, elem)
        return self.data
