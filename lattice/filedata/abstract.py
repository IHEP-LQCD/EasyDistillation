import abc
from typing import Any, List, Tuple


class FileMetaData:
    def __init__(self, shape: List[int], dtype: str = "<c16", extra: Any = None):
        self.shape = shape
        self.dtype = dtype
        self.extra = extra


class FileData(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __getitem__(self, key: Tuple[int]):
        pass


class File(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def getFileData(self, name: str, elem: FileMetaData) -> FileData:
        pass
