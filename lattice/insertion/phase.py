from typing import Tuple

from ..backend import get_backend


class MomentumPhase:
    def __init__(self, Lx: int, Ly: int, Lz: int) -> None:
        backend = get_backend()
        self.x = backend.arange(Lx).reshape(1, 1, Lx).repeat(Lz, 0).repeat(Ly, 1) * 2j * backend.pi / Lx
        self.y = backend.arange(Ly).reshape(1, Ly, 1).repeat(Lz, 0).repeat(Lx, 2) * 2j * backend.pi / Ly
        self.z = backend.arange(Lz).reshape(Lz, 1, 1).repeat(Ly, 1).repeat(Lx, 2) * 2j * backend.pi / Lz
        self.cache = {}

    def get(self, np: Tuple[int]):
        npx, npy, npz = np
        if np not in self.cache:
            backend = get_backend()
            self.cache[np] = backend.exp(npx * self.x + npy * self.y + npz * self.z)
        return self.cache[np]
