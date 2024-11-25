from typing import List, Tuple

from opt_einsum import contract

from ..constant import Nc, Nd
from ..backend import get_backend
from ..preset import GaugeField, Eigenvector
from ..insertion.phase import MomentumPhase


class DisplacementElementalGenerator:
    def __init__(
        self,
        latt_size: List[int],
        gauge_field: GaugeField,
        eigenvector: Eigenvector,
        distance: int = 0,
        momentum_list: List[Tuple[int]] = [(0, 0, 0)],
    ) -> None:
        backend = get_backend()
        Lx, Ly, Lz, Lt = latt_size
        self.kernel = None
        if backend.__name__ == "cupy":
            import os

            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stout_smear.cu")) as f:
                code = f.read()
            self.kernel = backend.RawModule(
                code=code, options=("--std=c++11",), name_expressions=("stout_smear<double>",)
            ).get_function(
                "stout_smear<double>"
            )  # TODO: More template instance.

        self.latt_size = latt_size
        self.gauge_field = gauge_field
        self.eigenvector = eigenvector
        self.momentum_list = momentum_list
        self.num_momentum = len(momentum_list)
        Ne = eigenvector.Ne
        self.Ne = eigenvector.Ne

        # self._U = backend.zeros((Nd, Lz * Ly * Lx, Nc, Nc), "<c16")
        self._U = None
        self._V = backend.zeros((Ne, Lz, Ly, Lx, Nc), "<c8")
        self._VPV = backend.zeros((distance + 1, self.num_momentum, Ne, Ne), "<c16")
        self._gauge_field_data = None
        self._eigenvector_data = None
        self._momentum_phase = MomentumPhase(latt_size)

        self.distance = distance
        self.Vd = backend.zeros((2 * (Nd - 1), Ne, Lz, Ly, Lx, Nc), "<c16")

    def _D(self, V, U, distance):
        backend = get_backend()
        Vd = self.Vd
        if distance == 0:
            return V
        elif distance == 1:
            for d in range(Nd - 1):
                Vf = backend.roll(V, -1, 3 - d)
                Vd[d] = contract("zyxab,ezyxb->ezyxa", U[d], Vf)
                UdV = contract("zyxba,ezyxb->ezyxa", U[d].conj(), V)
                Vd[-d - 1] = backend.roll(UdV, 1, 3 - d)
            return Vd.mean(0)
        else:
            for d in range(Nd - 1):
                Vf = backend.roll(Vd[d], -1, 3 - d)
                Vd[d] = contract("zyxab,ezyxb->ezyxa", U[d], Vf)
                UdV = contract("zyxba,ezyxb->ezyxa", U[d].conj(), Vd[-d - 1])
                Vd[-d - 1] = backend.roll(UdV, 1, 3 - d)
            return Vd.mean(0)

    def load(self, key: str):
        self._U = self.gauge_field.load(key)[:].transpose(4, 0, 1, 2, 3, 5, 6)[: Nd - 1]
        self._gauge_field_data = self.gauge_field.load(key).file
        self._eigenvector_data = self.eigenvector.load(key)

    def calc(self, t: int):
        gauge_field = self._gauge_field_data
        eigenvector = self._eigenvector_data
        momentum_phase = self._momentum_phase
        U = self._U
        V = self._V
        VPV = self._VPV
        # if self.distance > 0:
        #     for d in range(U.shape[0]):
        #         U[d] = gauge_field[t, :, d]
        for e in range(V.shape[0]):
            V[e] = eigenvector[t, e]
        for dist in range(self.distance + 1):
            VPV[dist] = 0
            right = self._D(V, U[:, t], dist)
            left = V
            for imom, mom in enumerate(self.momentum_list):
                VPV[dist, imom] += contract("zyx,ezyxc,fzyxc->ef", momentum_phase.get(mom), left.conj(), right)
        return VPV

    def project_SU3(self):
        backend = get_backend()
        U = self._U
        Uinv = backend.linalg.inv(U)
        while (
            backend.max(backend.abs(U - contract("...ab->...ba", Uinv.conj()))) > 1e-15
            or backend.max(backend.abs(contract("...ab,...cb", U, U.conj()) - backend.identity(Nc))) > 1e-15
        ):
            U = 0.5 * (U + contract("...ab->...ba", Uinv.conj()))
            Uinv = backend.linalg.inv(U)
        self._U = U

    def _stout_smear_ndarray(self, nstep, rho):
        backend = get_backend()
        U = backend.ascontiguousarray(self._U)

        for _ in range(nstep):
            Q = backend.zeros_like(U)
            U_dag = U.transpose(0, 1, 2, 3, 4, 6, 5).conj()
            for mu in range(Nd - 1):
                for nu in range(Nd - 1):
                    if mu != nu:
                        Q[mu] += U[nu] @ backend.roll(U[mu], -1, 3 - nu) @ backend.roll(U_dag[nu], -1, 3 - mu)
                        Q[mu] += (
                            backend.roll(U_dag[nu], +1, 3 - nu)
                            @ backend.roll(U[mu], +1, 3 - nu)
                            @ backend.roll(backend.roll(U[nu], +1, 3 - nu), -1, 3 - mu)
                        )

            Q = rho * Q @ U_dag
            Q = 0.5j * (Q.transpose(0, 1, 2, 3, 4, 6, 5).conj() - Q)
            contract("...aa->...a", Q)[:] -= 1 / Nc * contract("...aa->...", Q)[..., None]
            Q_sq = Q @ Q
            c0 = contract("...aa->...", Q @ Q_sq).real / 3
            c1 = contract("...aa->...", Q_sq).real / 2
            c0_max = 2 * (c1 / 3) ** (3 / 2)
            parity = c0 < 0
            c0 = backend.abs(c0)
            theta = backend.arccos(c0 / c0_max)
            u = (c1 / 3) ** 0.5 * backend.cos(theta / 3)
            w = c1**0.5 * backend.sin(theta / 3)
            u_sq = u**2
            w_sq = w**2
            e_iu_real = backend.cos(u)
            e_iu_imag = backend.sin(u)
            e_2iu_real = backend.cos(2 * u)
            e_2iu_imag = backend.sin(2 * u)
            cos_w = backend.cos(w)
            sinc_w = 1 - w_sq / 6 * (1 - w_sq / 20 * (1 - w_sq / 42 * (1 - w_sq / 72)))
            large = backend.abs(w) > 0.05
            w_large = w[large]
            sinc_w[large] = backend.sin(w_large) / w_large
            f_denom = 1 / (9 * u_sq - w_sq)
            f0_real = (
                (u_sq - w_sq) * e_2iu_real
                + e_iu_real * 8 * u_sq * cos_w
                + e_iu_imag * 2 * u * (3 * u_sq + w_sq) * sinc_w
            ) * f_denom
            f0_imag = (
                (u_sq - w_sq) * e_2iu_imag
                - e_iu_imag * 8 * u_sq * cos_w
                + e_iu_real * 2 * u * (3 * u_sq + w_sq) * sinc_w
            ) * f_denom
            f1_real = (
                2 * u * e_2iu_real - e_iu_real * 2 * u * cos_w + e_iu_imag * (3 * u_sq - w_sq) * sinc_w
            ) * f_denom
            f1_imag = (
                2 * u * e_2iu_imag + e_iu_imag * 2 * u * cos_w + e_iu_real * (3 * u_sq - w_sq) * sinc_w
            ) * f_denom
            f2_real = (e_2iu_real - e_iu_real * cos_w - e_iu_imag * 3 * u * sinc_w) * f_denom
            f2_imag = (e_2iu_imag + e_iu_imag * cos_w - e_iu_real * 3 * u * sinc_w) * f_denom
            f0_imag[parity] *= -1
            f1_real[parity] *= -1
            f2_imag[parity] *= -1

            f = (f2_real + 1j * f2_imag)[..., None, None] * Q_sq
            f += (f1_real + 1j * f1_imag)[..., None, None] * Q
            contract("...aa->...a", f)[:] += (f0_real + 1j * f0_imag)[..., None]
            U = f @ U
        self._U = U

    def _stout_smear_cuda_kernel(self, nstep, rho):
        backend = get_backend()
        Lx, Ly, Lz, Lt = self.latt_size
        U = backend.ascontiguousarray(self._U)

        for _ in range(nstep):
            U_in = U.copy()
            self.kernel((Lx * Ly * Lz, Nd - 1, 1), (Lt, 1, 1), (U, U_in, rho, Lx, Ly, Lz, Lt))

        self._U = U

    def _stout_smear_quda(self, nstep, rho):
        backend = get_backend()
        from pyquda_utils import io

        gauge = io.readQIOGauge(self._gauge_field_path)
        latt_size = gauge.latt_size
        Lx, Ly, Lz, Lt = latt_size

        gauge.smearSTOUT(nstep, rho, dir_ignore=3)

        self._U = backend.asarray(gauge.lexico()[: Nd - 1])

    def stout_smear(self, nstep, rho):
        from ..backend import check_QUDA

        backend = get_backend()
        if backend.__name__ == "numpy":
            self._stout_smear_ndarray(nstep, rho)
        elif backend.__name__ == "cupy":
            if self.kernel is not None:
                self._stout_smear_cuda_kernel(nstep, rho)
            elif check_QUDA():
                self._stout_smear_quda(nstep, rho)
            else:
                self._stout_smear_ndarray(nstep, rho)
