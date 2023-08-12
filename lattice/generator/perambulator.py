from typing import List

from opt_einsum import contract

from ..constant import Nc, Ns
from ..backend import set_backend, get_backend, check_QUDA
from ..preset import GaugeField, Eigenvector


class PerambulatorGenerator:  # TODO: Add parameters to do smearing before the inversion.
    def __init__(
        self,
        latt_size: List[int],
        gauge_field: GaugeField,
        eigenvector: Eigenvector,
        mass: float,
        tol: float,
        maxiter: int,
        xi_0: float = 1.0,
        nu: float = 1.0,
        clover_coeff_t: float = 0.0,
        clover_coeff_r: float = 1.0,
        anti_periodic_t: bool = True,
        multigrid: List[List[int]] = None,
    ) -> None:
        if not check_QUDA():
            raise ImportError("Please install PyQuda to generate the perambulator")
        from pyquda import core

        backend = get_backend()
        assert (
            backend.__name__ == "cupy"
        ), "PyQuda only support cupy as the ndarray implementation"
        Lx, Ly, Lz, Lt = latt_size
        Ne = eigenvector.Ne

        self.latt_size = latt_size
        self.gauge_field = gauge_field
        self.eigenvector = eigenvector
        self.dslash = core.getDslash(
            latt_size,
            mass,
            tol,
            maxiter,
            xi_0,
            nu,
            clover_coeff_t,
            clover_coeff_r,
            anti_periodic_t,
            multigrid,
        )
        self._SV = backend.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc), "<c16")
        self._VSV = backend.zeros((Lt, Ns, Ns, Ne, Ne), "<c16")

    def load(self, key: str):
        import numpy as np
        from pyquda import core
        from pyquda.utils import gauge_utils

        backend = get_backend()
        set_backend("numpy")
        Lx, Ly, Lz, Lt = self.latt_size
        Ne = self.eigenvector.Ne
        self.dslash.loadGauge(gauge_utils.readIldg(self.gauge_field.load(key).file))
        eigenvector_data = self.eigenvector.load(key)
        eigenvector_data_cb2 = np.zeros((Ne, Lt, Lz, Ly, Lx, Nc), "<c16")
        for e in range(Ne):
            for t in range(Lt):
                eigenvector_data_cb2[e, t] = eigenvector_data[t, e]
        eigenvector_data_cb2 = backend.asarray(
            core.cb2(eigenvector_data_cb2.reshape(Ne, Lt, Lz, Ly, Lx, Nc), [1, 2, 3, 4])
        )
        self._eigenvector_data = eigenvector_data_cb2
        set_backend(backend)

    def calc(self, t: int):
        from pyquda.field import LatticeFermion

        latt_size = self.latt_size
        Lx, Ly, Lz, Lt = latt_size
        Vol = Lx * Ly * Lz * Lt
        Ne = self.eigenvector.Ne
        eigenvector = self._eigenvector_data
        dslash = self.dslash

        SV = self._SV
        VSV = self._VSV

        for eigen in range(Ne):
            for spin in range(Ns):
                V = LatticeFermion(latt_size)
                data = V.data.reshape(2, Lt, Lz, Ly, Lx // 2, Ns, Nc)
                data[:, t, :, :, :, spin, :] = eigenvector[eigen, :, t, :, :, :, :]
                SV.reshape(Vol, Ns, Ns, Nc)[:, :, spin, :] = dslash.invert(
                    V
                ).data.reshape(Vol, Ns, Nc)
            VSV[:, :, :, :, eigen] = contract(
                "ketzyxa,etzyxija->tijk", eigenvector.conj(), SV
            )
        return VSV
