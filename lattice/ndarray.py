from time import time
from typing import Tuple

from .abstract import ElementMetaData, FileData, File
from .backend import getBackend, getNumpy


class NdarrayFileData(FileData):
    def __init__(self, file: str, elem: ElementMetaData) -> None:
        self.file = file
        self.timeInSec = 0.0
        self.sizeInByte = 0

    def __getitem__(self, key: Tuple[int]):
        numpy = getBackend()
        numpy_ori = getNumpy()
        s = time()
        ret = numpy.asarray(numpy_ori.load(self.file, "r")[key])
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
