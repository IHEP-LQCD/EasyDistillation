from io import FileIO
import os
import random
from time import sleep
from typing import Iterator, Union

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


def rand(seed: Union[int, float, str, bytes, bytearray] = None) -> str:
    if seed is None:
        ret = 0x456789AB
    else:
        random.seed(seed)
        ret = random.randint(0, 0xFFFFFFFF)
    return hex(ret)[2:].upper()


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


class Dispatch:
    def __init__(self, input: str, seed: Union[int, float, str, bytes, bytearray] = None) -> None:
        tmp = f"{input}.{rand(seed=seed)}.tmp"
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
        from .process import processBar

        with open(input, "r") as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines if line != ""]
        return processBar(lines)


def combine(output: str, line: str):
    with AtomicOpen(output, "a") as f:
        f.write(f"{line}\n")
