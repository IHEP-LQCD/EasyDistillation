from io import FileIO
import os
import random
from time import sleep
from typing import Iterator, Iterable, TypeVar

T = TypeVar("T")

if os.name == "nt":
    import msvcrt

    def lock(f: FileIO):
        while True:
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBRLCK, 1)
                break
            except OSError:
                sleep(0.1)

    def unlock(f: FileIO):
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)

elif os.name == "posix":
    import fcntl

    def lock(f: FileIO):
        if f.writable():
            fcntl.lockf(f, fcntl.LOCK_EX)

    def unlock(f: FileIO):
        if f.writable():
            fcntl.lockf(f, fcntl.LOCK_UN)


class AtomicOpen:
    def __init__(self, path, *args, **kwargs):
        self.file = open(path, *args, **kwargs)
        self.begin = self.file.tell()
        lock(self.file)

    def __enter__(self, *args, **kwargs):
        return self.file

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.seek(self.begin)
        unlock(self.file)
        self.file.close()
        if exc_type is not None:
            return False
        else:
            return True


def processBar(input: Iterable[T], length: int = 100, filled: str = "#", empty: str = "-") -> Iterator[T]:
    total = len(input)
    i = 0
    for j in input:
        percentage = i / total
        num = int(percentage * length)
        print(f"\r[{filled * num + empty * (length - num)}] {percentage * 100 : 6.2f}%", end=" ")
        yield j
        i += 1
    print(f"\r[{filled * length}] {100.0 : 6.2f}%")


def rand(suffix: str = None) -> str:
    ret = str(suffix)
    if suffix is None:
        ret = hex(random.randint(0, 0xFFFFFFFF))[2:].upper()
    return ret


class Dispatch:
    def __init__(self, input: str, suffix: str = None) -> None:
        tmp = f"{input}.{rand(suffix=suffix)}.tmp"
        self.tmp = tmp
        try:
            with AtomicOpen(tmp, "x+") as f:
                with open(input, "r") as fi:
                    f.write(fi.read())
        except FileExistsError:
            pass

    def __iter__(self) -> Iterator[str]:
        while True:
            with AtomicOpen(self.tmp, "r+") as f:
                lines = f.readlines()
                f.seek(0)
                if lines == []:
                    line = None
                else:
                    line = lines.pop(0).strip()
                    f.writelines(lines)
                    f.truncate()
            if line is None:
                break
            elif line == "":
                continue
            else:
                yield line

    @staticmethod
    def process(input: str):
        with open(input, "r") as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines if line != ""]
        return processBar(lines)


def combine(output: str, line: str):
    with AtomicOpen(output, "a") as f:
        f.write(f"{line}\n")
