import abc
from typing import Any, List, Tuple


class FileMetaData:
    def __init__(self, shape: List[int], dtype: str = "<c16", extra: Any = None):
        self.shape = shape
        self.dtype = dtype
        self.extra = extra


class FileData(metaclass=abc.ABCMeta):
    shape = None
    dtype = None
    time_in_sec = 0.0
    size_in_byte = 0

    @abc.abstractmethod
    def __getitem__(self, key: Tuple[int]):
        pass


class File(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_file_data(self, name: str, elem: FileMetaData) -> FileData:
        pass
