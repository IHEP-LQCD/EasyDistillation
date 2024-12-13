import functools

from opt_einsum import contract

from ..constant import Nc, Nd
from ..backend import get_backend, check_QUDA


def _Laplacian(F, U, U_dag, latt_size):
    backend = get_backend()
    Lx, Ly, Lz, Lt = latt_size
    F = F.reshape(Lz, Ly, Lx, Nc, -1)
    return (
        # - for SA with evals , + for LA with (12 - evals)
        6 * F
        - (
            contract("zyxab,zyxbc->zyxac", U[0], backend.roll(F, -1, 2))
            + contract("zyxab,zyxbc->zyxac", U[1], backend.roll(F, -1, 1))
            + contract("zyxab,zyxbc->zyxac", U[2], backend.roll(F, -1, 0))
            + backend.roll(contract("zyxab,zyxbc->zyxac", U_dag[0], F), 1, 2)
            + backend.roll(contract("zyxab,zyxbc->zyxac", U_dag[1], F), 1, 1)
            + backend.roll(contract("zyxab,zyxbc->zyxac", U_dag[2], F), 1, 0)
        )
    ).reshape(Lz * Ly * Lx * Nc, -1)


class EigenvectorGenerator:
    def __init__(self, latt_size, gauge_field, Ne, tol) -> None:
        backend = get_backend()
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
        self.Ne = Ne
        self.tol = tol
        self._U = None

    def load(self, key: str):
        self._U = self.gauge_field.load(key)[:].transpose(4, 0, 1, 2, 3, 5, 6).copy()
        print(f"{self.gauge_field.load(key).sizeInByte/1024**2/self.gauge_field.load(key).timeInSec:.3f} MB/s")
        self._gauge_field_path = self.gauge_field.load(key).file

    def project_SU3(self):
        backend = get_backend()
        U = self._U[: Nd - 1]
        Uinv = backend.linalg.inv(U)
        while (
            backend.max(backend.abs(U - contract("...ab->...ba", Uinv.conj()))) > 1e-15
            or backend.max(backend.abs(contract("...ab,...cb", U, U.conj()) - backend.identity(Nc))) > 1e-15
        ):
            U = 0.5 * (U + contract("...ab->...ba", Uinv.conj()))
            Uinv = backend.linalg.inv(U)
        self._U[: Nd - 1] = U

    def _stout_smear_ndarray_naive(self, nstep, rho):
        backend = get_backend()
        U = backend.ascontiguousarray(self._U[: Nd - 1])

        for _ in range(nstep):
            Q = backend.zeros_like(U)
            for mu in range(Nd - 1):
                for nu in range(Nd - 1):
                    if mu != nu:
                        Q[mu] += contract(
                            "...ab,...bc,...dc->...ad",
                            U[nu],
                            backend.roll(U[mu], -1, 3 - nu),
                            backend.roll(U[nu], -1, 3 - mu).conj(),
                        )
                        Q[mu] += contract(
                            "...ba,...bc,...cd->...ad",
                            backend.roll(U[nu], +1, 3 - nu).conj(),
                            backend.roll(U[mu], +1, 3 - nu),
                            backend.roll(backend.roll(U[nu], +1, 3 - nu), -1, 3 - mu),
                        )
            Q = contract("...ab,...cb->...ac", rho * Q, U.conj())
            Q = 0.5j * (contract("...ab->...ba", Q.conj()) - Q)
            Q -= 1 / Nc * contract("...aa,bc->...bc", Q, backend.identity(Nc))
            c0 = contract("...ab,...bc,...ca->...", Q, Q, Q).real / 3
            c1 = contract("...ab,...ba->...", Q, Q).real / 2

            c0_max = 2 * (c1 / 3) ** (3 / 2)
            parity = c0 < 0
            c0 = backend.abs(c0)
            theta = backend.arccos(c0 / c0_max)
            u = (c1 / 3) ** 0.5 * backend.cos(theta / 3)
            w = c1**0.5 * backend.sin(theta / 3)
            u_sq = u**2
            w_sq = w**2
            e_iu = backend.exp(-1j * u)
            e_2iu = backend.exp(2j * u)
            cos_w = backend.cos(w)
            sinc_w = 1 - w_sq / 6 * (1 - w_sq / 20 * (1 - w_sq / 42 * (1 - w_sq / 72)))
            large = backend.abs(w) > 0.05
            w_large = w[large]
            sinc_w[large] = backend.sin(w_large) / w_large
            f_denom = 1 / (9 * u_sq - w_sq)
            f0 = ((u_sq - w_sq) * e_2iu + e_iu * (8 * u_sq * cos_w + 2j * u * (3 * u_sq + w_sq) * sinc_w)) * f_denom
            f1 = (2 * u * e_2iu - e_iu * (2 * u * cos_w - 1j * (3 * u_sq - w_sq) * sinc_w)) * f_denom
            f2 = (e_2iu - e_iu * (cos_w + 3j * u * sinc_w)) * f_denom
            f0[parity] = f0[parity].conj()
            f1[parity] = -f1[parity].conj()
            f2[parity] = f2[parity].conj()

            f0 = contract("...,ab->...ab", f0, backend.identity(Nc))
            f1 = contract("...,...ab->...ab", f1, Q)
            f2 = contract("...,...ab,...bc->...ac", f2, Q, Q)
            U = contract("...ab,...bc->...ac", f0 + f1 + f2, U)
        self._U[: Nd - 1] = U

    def _stout_smear_ndarray(self, nstep, rho):
        backend = get_backend()
        U = backend.ascontiguousarray(self._U[: Nd - 1])

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
        self._U[: Nd - 1] = U

    def _stout_smear_cuda_kernel(self, nstep, rho):
        backend = get_backend()
        Lx, Ly, Lz, Lt = self.latt_size
        U = backend.ascontiguousarray(self._U[: Nd - 1])

        for _ in range(nstep):
            U_in = U.copy()
            self.kernel((Lx * Ly * Lz, Nd - 1, 1), (Lt, 1, 1), (U, U_in, rho, Lx, Ly, Lz, Lt))

        self._U[: Nd - 1] = U

    def _stout_smear_quda(self, nstep, rho):
        from pyquda_utils import io

        gauge = io.readQIOGauge(self._gauge_field_path)
        Lx, Ly, Lz, Lt = gauge.latt_info.size

        gauge.smearSTOUT(nstep, rho, dir_ignore=3)

        self.gauge_quda = gauge
        backend = get_backend()
        self._U = backend.asarray(gauge.lexico().reshape(Nd, Lt, Lz, Ly, Lx, Nc, Nc))

    def stout_smear(self, nstep, rho):
        from ..backend import check_QUDA

        backend = get_backend()
        if backend.__name__ == "numpy":
            print(f"Use numpy to stout_smear(nstep={nstep}, rho={rho})")
            self._stout_smear_ndarray(nstep, rho)
        elif backend.__name__ == "cupy":
            if check_QUDA():
                print(f"Use quda to stout_smear(nstep={nstep}, rho={rho})")
                self._stout_smear_quda(nstep, rho)
            elif self.kernel is not None:
                print(f"Use cuda kernel to stout_smear(nstep={nstep}, rho={rho})")
                self._stout_smear_cuda_kernel(nstep, rho)
            else:
                print(f"Use cupy to stout_smear(nstep={nstep}, rho={rho})")
                self._stout_smear_ndarray(nstep, rho)
                # self._stout_smear_ndarray_naive(nstep, rho)

    def laplacian_cupy_numpy(self, t: int, apply_renorm_phase: bool):
        backend = get_backend()
        if backend.__name__ == "numpy":
            from scipy.sparse import linalg
        elif backend.__name__ == "cupy":
            from cupyx.scipy.sparse import linalg
        Lx, Ly, Lz, Lt = self.latt_size

        U = backend.asarray(self._U[: Nd - 1, t])
        U_dag = U.transpose(0, 1, 2, 3, 5, 4).conj()
        Laplacian = functools.partial(_Laplacian, U=U, U_dag=U_dag, latt_size=self.latt_size)
        A = linalg.LinearOperator((Lz * Ly * Lx * Nc, Lz * Ly * Lx * Nc), matvec=Laplacian, matmat=Laplacian)
        evals, evecs = linalg.eigsh(A, self.Ne, which="SA", tol=self.tol)
        evecs = evecs.transpose(1, 0).reshape(self.Ne, -1)

        # sort eigenvalues additionally
        argsort = backend.argsort(evals)
        evals, evecs = evals[argsort], evecs[argsort]

        # [Ne, Lz * Ly * Lx, Nc]
        if apply_renorm_phase:
            renorm_phase = backend.angle(evecs[:, 0])
            evecs *= backend.exp(-1.0j * renorm_phase)[:, None]
        return evecs.reshape(self.Ne, Lz, Ly, Lx, Nc), evals

    def laplacian_quda(self, t: int, apply_renorm_phase: bool):
        from cupyx.scipy.sparse import linalg
        from pyquda.field import (
            Nc,
            LatticeGauge,
            LatticeInfo,
            LatticeStaggeredFermion,
            cb2,
            lexico,
        )

        backend = get_backend()
        Lx, Ly, Lz, Lt = self.latt_size
        latt_info = LatticeInfo([Lx, Ly, Lz, 1])
        Lx, Ly, Lz, Lt = latt_info.size
        gauge_tmp = LatticeGauge(latt_info, backend.asarray(cb2(self._U[:, t : t + 1].get(), [1, 2, 3, 4])))
        gauge_tmp.ensurePureGauge()
        gauge_tmp.pure_gauge.loadGauge(gauge_tmp)

        def Laplacian(x):
            x = x.reshape(Lz * Ly * Lx * Nc, -1)
            b = backend.zeros_like(x, "<c16")
            for i in range(x.shape[1]):
                b[:, i] = gauge_tmp.pure_gauge.laplace(LatticeStaggeredFermion(latt_info, x[:, i]), 3).data.reshape(
                    Lz * Ly * Lx * Nc
                )
            return b

        A = linalg.LinearOperator((Lz * Ly * Lx * Nc, Lz * Ly * Lx * Nc), matvec=Laplacian, matmat=Laplacian)
        evals, evecs = linalg.eigsh(A, self.Ne, which="SA", tol=self.tol)
        evals *= 2 * (Nd - 1)
        evecs = evecs.transpose(1, 0).reshape(self.Ne, -1)

        gauge_tmp.pure_gauge.freeGauge()

        # sort eigenvalues additionally
        argsort = backend.argsort(evals)
        evals, evecs = evals[argsort], evecs[argsort]

        # [Ne, Lz * Ly * Lx, Nc]
        if apply_renorm_phase:
            renorm_phase = backend.angle(evecs[:, 0])
            evecs *= backend.exp(-1.0j * renorm_phase)[:, None]
        evecs = evecs.get()
        for i in range(self.Ne):
            evecs[i] = lexico(evecs[i].reshape(2, 1, Lz, Ly, Lx // 2, Nc), [0, 1, 2, 3, 4]).reshape(-1)
        evecs = backend.asarray(evecs)
        return evecs.reshape(self.Ne, Lz, Ly, Lx, Nc), evals

    def calc(self, t: int, apply_renorm_phase: bool = True):
        backend = get_backend()
        # Don't use QUDA's eigensolver because of some performance regression.
        if backend.__name__ == "cupy" and check_QUDA():
            print(f"Using quda Laplacian and {backend.__name__} solver.")
            return self.laplacian_quda(t, apply_renorm_phase)
        elif backend.__name__ == "cupy" or backend.__name__ == "numpy":
            print(f"Using {backend.__name__} Laplacian and {backend.__name__} solver.")
            return self.laplacian_cupy_numpy(t, apply_renorm_phase)
        else:
            raise NotImplementedError(f"Unsupport backend = {backend.__name__}.")
