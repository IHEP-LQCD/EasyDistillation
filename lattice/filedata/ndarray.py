import mmap
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
        # ret = numpy.asarray(
        #     loader(
        #         self.file,
        #     )[key]
        # )
        # ret = backend.asarray(numpy.load(self.file, mmap_mode="r")[key].copy())
        with open(self.file, "rb") as f:
            N = len(numpy.lib.format.MAGIC_PREFIX) + 2
            magic = f.read(N)
            assert magic[:-2] == numpy.lib.format.MAGIC_PREFIX
            major, minor = magic[-2:]
            version = (major, minor)
            assert version in [(1, 0), (2, 0)]
            shape, fortran_order, dtype = numpy.lib.format._read_array_header(f, version)
            assert not fortran_order
            self_offset = f.tell()
            start = self_offset - self_offset % mmap.ALLOCATIONGRANULARITY
            offset = self_offset - start
            with mmap.mmap(
                f.fileno(), offset + int(numpy.prod(shape)) * dtype.itemsize, access=mmap.ACCESS_READ, offset=start
            ) as mm:
                file = numpy.ndarray.__new__(numpy.memmap, shape=tuple(shape), dtype=dtype, buffer=mm, offset=offset)
                ret = backend.asarray(file[key].copy())
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


## the following modifications are for timeslide save solely data, the timeslice index must be the first.
class NdarrayTimeslicesFileData(FileData):
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

        tsrc_idx, *rest = key
        key = tuple(rest)
        import re

        self.file = re.sub(r"\.t\?\?\?\.", f".t{tsrc_idx:03d}.", self.file)

        with open(self.file, "rb") as f:
            N = len(numpy.lib.format.MAGIC_PREFIX) + 2
            magic = f.read(N)
            assert magic[:-2] == numpy.lib.format.MAGIC_PREFIX
            major, minor = magic[-2:]
            version = (major, minor)
            assert version in [(1, 0), (2, 0)]
            shape, fortran_order, dtype = numpy.lib.format._read_array_header(f, version)
            assert not fortran_order
            self_offset = f.tell()
            start = self_offset - self_offset % mmap.ALLOCATIONGRANULARITY
            offset = self_offset - start
            with mmap.mmap(
                f.fileno(), offset + int(numpy.prod(shape)) * dtype.itemsize, access=mmap.ACCESS_READ, offset=start
            ) as mm:
                file = numpy.ndarray.__new__(numpy.memmap, shape=tuple(shape), dtype=dtype, buffer=mm, offset=offset)
                ret = backend.asarray(file[key].copy())
        self.time_in_sec += time() - s
        self.size_in_byte += ret.nbytes
        return ret


class NdarrayTimeslicesFile(File):
    def __init__(self) -> None:
        self.file: str = None
        self.data: NdarrayFileData = None

    def get_file_data(self, name: str, elem: FileMetaData) -> NdarrayFileData:
        if self.file != name:
            self.file = name
            self.data = NdarrayTimeslicesFileData(name, elem)
        return self.data
