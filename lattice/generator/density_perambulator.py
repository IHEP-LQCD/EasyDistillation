from typing import List, Literal, Tuple

from opt_einsum import contract

from ..constant import Nc, Ns
from ..backend import set_backend, get_backend, check_QUDA
from ..preset import GaugeField, Eigenvector
from ..insertion.gamma import gamma
from ..insertion.phase import MomentumPhase


class DensityPerambulatorGenerator:  # TODO: Add parameters to do smearing before the inversion.
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
        t_boundary: Literal[1, -1] = 1,
        multigrid: List[List[int]] = None,
        gamma_list: List[int] = [i for i in range(Ns * Ns)],
        momentum_list: List[Tuple[int]] = [(0, 0, 0)],
    ) -> None:
        if not check_QUDA():
            raise ImportError("Please install PyQuda to generate the perambulator or check MPI_init again.")
        from pyquda import core
        from pyquda.field import LatticeInfo

        self.latt_info = LatticeInfo(latt_size=latt_size, t_boundary=t_boundary, anisotropy=xi_0 / nu)

        backend = get_backend()
        assert backend.__name__ == "cupy", "PyQuda only support cupy as the ndarray implementation"
        import numpy as np
        from cupyx import zeros_pinned

        Lx, Ly, Lz, Lt = latt_size
        Ne = eigenvector.Ne

        self.latt_size = latt_size
        self.gauge_field = gauge_field
        self.eigenvector = eigenvector
        # self.dslash = core.getDslash(
        #     latt_size, mass, tol, maxiter, xi_0, nu, clover_coeff_t, clover_coeff_r, anti_periodic_t, multigrid
        # ) # deprecated
        self.dirac = core.getDirac(
            self.latt_info,
            mass,
            tol,
            maxiter,
            xi_0,
            clover_coeff_t,
            clover_coeff_r,
            multigrid,
        )

        self.gamma_list = gamma_list
        self.momentum_list = momentum_list
        self._momentum_phase = MomentumPhase(latt_size)
        self._SV_i = backend.zeros((Ne, 2, Lz, Ly, Lx // 2, Ns, Ns, Nc), "<c16")
        self._SV_f = backend.zeros((Ne, 2, Lz, Ly, Lx // 2, Ns, Ns, Nc), "<c16")
        self._VSSV_cb2 = zeros_pinned((Ne, len(gamma_list), len(momentum_list), 2, Lz, Ly, Lx // 2, Ns, Ns), "<c16")
        self._VSSV = np.zeros((Ne, Ne, len(gamma_list), len(momentum_list), Lz, Ly, Lx, Ns, Ns), "<c16")
        self._stream = backend.cuda.Stream()
        self._t = None
        self._tf = None

    def load(self, key: str):
        from pyquda.utils import io

        self.dirac.loadGauge(io.readQIOGauge(self.gauge_field.load(key).file))
        self._eigenvector_data = self.eigenvector.load(key)

    def calc(self, ti: int, tf: int, tau: int):
        import numpy as np
        from pyquda.field import LatticeFermion

        backend = get_backend()
        latt_size = self.latt_size
        Lx, Ly, Lz, Lt = latt_size
        Ne = self.eigenvector.Ne
        eigenvector = self._eigenvector_data
        dirac = self.dirac
        gamma_list = self.gamma_list
        momentum_list = self.momentum_list
        momentum_phase = self._momentum_phase

        data_lexico = np.zeros((2, Ne, Lz, Ly, Lx, Nc), "<c16")
        data_cb2 = np.zeros((2, Ne, 2, Lz, Ly, Lx // 2, Nc), "<c16")
        set_backend("numpy")
        for e in range(Ne):
            if ti != self._t:
                data_lexico[0, e] = eigenvector[ti, e]
            if tf != self._tf:
                data_lexico[1, e] = eigenvector[tf, e]
        set_backend(backend)

        for z in range(Lz):
            for y in range(Ly):
                if ti != self._t:
                    eo = (ti + z + y) % 2
                    if eo == 0:
                        data_cb2[0, :, 0, z, y, :] = data_lexico[0, :, z, y, 0::2]
                        data_cb2[0, :, 1, z, y, :] = data_lexico[0, :, z, y, 1::2]
                    else:
                        data_cb2[0, :, 0, z, y, :] = data_lexico[0, :, z, y, 1::2]
                        data_cb2[0, :, 1, z, y, :] = data_lexico[0, :, z, y, 0::2]

                if tf != self._tf:
                    eo = (tf + z + y) % 2
                    if eo == 0:
                        data_cb2[1, :, 0, z, y, :] = data_lexico[1, :, z, y, 0::2]
                        data_cb2[1, :, 1, z, y, :] = data_lexico[1, :, z, y, 1::2]
                    else:
                        data_cb2[1, :, 0, z, y, :] = data_lexico[1, :, z, y, 1::2]
                        data_cb2[1, :, 1, z, y, :] = data_lexico[1, :, z, y, 0::2]
        data_cb2 = backend.asarray(data_cb2)

        _V = LatticeFermion(latt_size)
        V = _V.data.reshape(2, Lt, Lz, Ly, Lx // 2, Ns, Nc)
        SV_i = self._SV_i
        SV_f = self._SV_f
        VSSV_cb2 = self._VSSV_cb2
        VSSV = self._VSSV
        stream = self._stream

        for eigen in range(Ne):
            if ti != self._t:
                for spin in range(Ns):
                    V[:, ti, :, :, :, spin, :] = data_cb2[0, eigen, :, :, :, :, :]
                    SV_i[eigen, :, :, :, :, :, spin, :] = dirac.invert(_V).data.reshape(
                        2, Lt, Lz, Ly, Lx // 2, Ns, Nc
                    )[:, tau, :, :, :, :, :]
                    V[:] = 0

            if tf != self._tf:
                for spin in range(Ns):
                    V[:, tf, :, :, :, spin, :] = data_cb2[1, eigen, :, :, :, :, :]
                    SV_f[eigen, :, :, :, :, :, spin, :] = dirac.invert(_V).data.reshape(
                        2, Lt, Lz, Ly, Lx // 2, Ns, Nc
                    )[:, tau, :, :, :, :, :]
                    V[:] = 0
            SV_f[:] = contract("ii,kezyxjic,jj->kezyxijc", gamma(15), SV_f.conj(), gamma(15))

        if ti != self._t:
            self._t = ti
        if tf != self._tf:
            self._tf = tf

        for eigen_f in range(Ne):
            for eigen_i in range(Ne):
                for gamma_idx, gamma_i in enumerate(gamma_list):
                    for momentum_idx, momentum in enumerate(momentum_list):
                        contract(
                            "ezyx,ezyxijc,jk,ezyxklc->ezyxil",
                            momentum_phase.get_cb2(momentum)[:, tau],
                            SV_f[eigen_f],
                            gamma(gamma_i),
                            SV_i[eigen_i],
                        ).get(stream, out=VSSV_cb2[eigen_i, gamma_idx, momentum_idx])
            stream.synchronize()
            for z in range(Lz):
                for y in range(Ly):
                    eo = (tau + z + y) % 2
                    if eo == 0:
                        VSSV[eigen_f, :, :, :, z, y, 1::2] = VSSV_cb2[:, :, :, 1, z, y, :]
                        VSSV[eigen_f, :, :, :, z, y, 0::2] = VSSV_cb2[:, :, :, 0, z, y, :]
                    else:
                        VSSV[eigen_f, :, :, :, z, y, 1::2] = VSSV_cb2[:, :, :, 0, z, y, :]
                        VSSV[eigen_f, :, :, :, z, y, 0::2] = VSSV_cb2[:, :, :, 1, z, y, :]

        return VSSV.transpose(2, 3, 4, 5, 6, 7, 8, 0, 1)
