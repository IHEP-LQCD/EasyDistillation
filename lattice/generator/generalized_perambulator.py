from typing import List, Literal, Tuple

from opt_einsum import contract
from time import perf_counter

from ..constant import Nc, Ns
from ..backend import set_backend, get_backend, check_QUDA
from ..preset import GaugeField, Eigenvector
from ..insertion.gamma import gamma
from ..insertion.phase import MomentumPhase


class GeneralizedPerambulatorGenerator:  # TODO: Add parameters to do smearing before the inversion.
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
        from cupyx import zeros_pinned

        Lx, Ly, Lz, Lt = latt_size
        Ne = eigenvector.Ne

        self.latt_size = latt_size
        self.gauge_field = gauge_field
        self.eigenvector = eigenvector
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
        self._SV_i = backend.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc), "<c16")
        self._SV_f = backend.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc), "<c16")
        self._h_SV_i = zeros_pinned((Ne, 2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc), "<c16")
        self._h_SV_f = zeros_pinned((Ne, 2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc), "<c16")
        self._VSSV_fi = backend.zeros((len(gamma_list), len(momentum_list), Lt, Ns, Ns), "<c16")
        self._VSSV = zeros_pinned((Ne, Ne, len(gamma_list), len(momentum_list), Lt, Ns, Ns), "<c16")
        self._stream_i = backend.cuda.Stream()
        self._stream_f = backend.cuda.Stream()
        self._ti = None
        self._tf = None

    def load(self, key: str):
        from pyquda.utils import io

        self.dirac.loadGauge(io.readQIOGauge(self.gauge_field.load(key).file))
        self._eigenvector_data = self.eigenvector.load(key)

    def calc(self, ti: int, tf: int):
        import numpy as np
        from pyquda.field import LatticeFermion

        s = perf_counter()

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
            if ti != self._ti:
                data_lexico[0, e] = eigenvector[ti, e]
            if tf != self._tf:
                data_lexico[1, e] = eigenvector[tf, e]
        set_backend(backend)
        data_lexico = data_lexico.reshape(2, Ne, Lz, Ly, Lx, Nc)

        for z in range(Lz):
            for y in range(Ly):
                if ti != self._ti:
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

        print(f"load lexico timer = {perf_counter() -s:.2f}")

        _V = LatticeFermion(self.latt_info)
        V = _V.data.reshape(2, Lt, Lz, Ly, Lx // 2, Ns, Nc)
        SV_i = self._SV_i
        SV_f = self._SV_f
        h_SV_i = self._h_SV_i
        h_SV_f = self._h_SV_f
        VSSV_fi = self._VSSV_fi
        VSSV = self._VSSV
        stream_i = self._stream_i
        stream_f = self._stream_f

        s = perf_counter()
        for eigen in range(Ne):
            print(f"calc: Ne = {eigen},, time = {s-perf_counter():.2f} sec.")
            if ti != self._ti:
                stream_i.synchronize()
                for spin in range(Ns):
                    V[:, ti, :, :, :, spin, :] = data_cb2[0, eigen, :, :, :, :, :]
                    SV_i[:, :, :, :, :, :, spin, :] = dirac.invert(_V).data.reshape(2, Lt, Lz, Ly, Lx // 2, Ns, Nc)
                    V[:] = 0
                SV_i.get(stream_i, out=h_SV_i[eigen])

            if tf != self._tf:
                stream_f.synchronize()
                for spin in range(Ns):
                    V[:, tf, :, :, :, spin, :] = data_cb2[1, eigen, :, :, :, :, :]
                    SV_f[:, :, :, :, :, :, spin, :] = dirac.invert(_V).data.reshape(2, Lt, Lz, Ly, Lx // 2, Ns, Nc)
                    V[:] = 0
                SV_f[:] = contract("ii,etzyxjic,jj->etzyxijc", gamma(15), SV_f.conj(), gamma(15))
                SV_f.get(stream_f, out=h_SV_f[eigen])

        if ti != self._ti:
            self._ti = ti
        if tf != self._tf:
            self._tf = tf
        stream_i.synchronize()
        stream_f.synchronize()

        s = perf_counter()
        for eigen_i in range(Ne):
            print(f"calc contraction: Ne = {eigen_i}, time = {s-perf_counter():.2f} sec.")
            SV_i.set(h_SV_i[eigen_i])
            for eigen_f in range(Ne):
                SV_f.set(h_SV_f[eigen_f])
                stream_i.synchronize()
                gamma_current = backend.asarray([gamma(i) for i in gamma_list])
                for momentum_idx, momentum in enumerate(momentum_list):
                    VSSV_fi[:, momentum_idx] = contract(
                        "etzyx,etzyxijc,gjk,etzyxklc->gtil",
                        momentum_phase.get_cb2(momentum),
                        SV_f,
                        gamma_current,
                        SV_i,
                        optimize=True,
                    )
                VSSV_fi.get(stream_i, out=VSSV[eigen_f, eigen_i])

        return VSSV.transpose(2, 3, 4, 5, 6, 0, 1)
