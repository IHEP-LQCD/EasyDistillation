import abc
from typing import Any, List, NamedTuple, Tuple


class FileMetaData(NamedTuple):
    shape: List[int]
    dtype: str
    extra: Any


class FileData(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __getitem__(self, key: Tuple[int]):
        pass


class File(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __getitem__(self, key: Tuple[int]):
        pass

    @abc.abstractmethod
    def getFileData(self, name: str, elem: FileMetaData) -> FileData:
        pass
