from typing import List, Tuple

from ..backend import get_backend


class MomentumPhase:
    def __init__(self, latt_size: List[int]) -> None:
        backend = get_backend()
        self.latt_size = latt_size
        Lx, Ly, Lz, Lt = self.latt_size
        self.x = backend.arange(Lx).reshape(1, 1, Lx).repeat(Lz, 0).repeat(Ly, 1) * 2j * backend.pi / Lx
        self.y = backend.arange(Ly).reshape(1, Ly, 1).repeat(Lz, 0).repeat(Lx, 2) * 2j * backend.pi / Ly
        self.z = backend.arange(Lz).reshape(Lz, 1, 1).repeat(Ly, 1).repeat(Lx, 2) * 2j * backend.pi / Lz
        x = self.x.reshape(1, Lz, Ly, Lx).repeat(Lt, 0)
        y = self.y.reshape(1, Lz, Ly, Lx).repeat(Lt, 0)
        z = self.z.reshape(1, Lz, Ly, Lx).repeat(Lt, 0)
        self.x_cb2 = backend.zeros((2, Lt, Lz, Ly, Lx // 2), "<c16")
        self.y_cb2 = backend.zeros((2, Lt, Lz, Ly, Lx // 2), "<c16")
        self.z_cb2 = backend.zeros((2, Lt, Lz, Ly, Lx // 2), "<c16")
        for it in range(Lt):
            for iz in range(Lz):
                for iy in range(Ly):
                    ieo = (it + iz + iy) % 2
                    if ieo == 0:
                        self.x_cb2[0, it, iz, iy] = x[it, iz, iy, 0::2]
                        self.x_cb2[1, it, iz, iy] = x[it, iz, iy, 1::2]
                        self.y_cb2[0, it, iz, iy] = y[it, iz, iy, 0::2]
                        self.y_cb2[1, it, iz, iy] = y[it, iz, iy, 1::2]
                        self.z_cb2[0, it, iz, iy] = z[it, iz, iy, 0::2]
                        self.z_cb2[1, it, iz, iy] = z[it, iz, iy, 1::2]
                    else:
                        self.x_cb2[1, it, iz, iy] = x[it, iz, iy, 0::2]
                        self.x_cb2[0, it, iz, iy] = x[it, iz, iy, 1::2]
                        self.y_cb2[1, it, iz, iy] = y[it, iz, iy, 0::2]
                        self.y_cb2[0, it, iz, iy] = y[it, iz, iy, 1::2]
                        self.z_cb2[1, it, iz, iy] = z[it, iz, iy, 0::2]
                        self.z_cb2[0, it, iz, iy] = z[it, iz, iy, 1::2]
        self.cache = {}
        self.cache_cb2 = {}

    def get(self, np: Tuple[int]):
        npx, npy, npz = np
        if np not in self.cache:
            backend = get_backend()
            self.cache[np] = backend.exp(npx * self.x + npy * self.y + npz * self.z)
        return self.cache[np]

    def get_cb2(self, np: Tuple[int]):
        npx, npy, npz = np
        if np not in self.cache_cb2:
            backend = get_backend()
            self.cache_cb2[np] = backend.exp(npx * self.x_cb2 + npy * self.y_cb2 + npz * self.z_cb2)
        return self.cache_cb2[np]
