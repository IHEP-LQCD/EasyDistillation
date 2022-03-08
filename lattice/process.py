from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


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
