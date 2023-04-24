import functools
from ..backend import set_backend, get_backend
if True:
    set_backend("numpy")
    backend = get_backend()
    from scipy.sparse import linalg
else:
    set_backend("cupy")
    backend = get_backend()
    from cupyx.scipy.sparse import linalg
from opt_einsum import contract

from ..constant import Nc, Nd


def _Laplacian(colvec, U, U_dag, latt_size):
    Lx, Ly, Lz, Lt = latt_size
    colvec = colvec.reshape(Lz, Ly, Lx, Nc, -1)
    return (
        # - for SA with evals , + for LA with (12 - evals)
        6 * colvec - (
            contract("zyxab,zyxbc->zyxac", U[0], backend.roll(colvec, -1, 2)) +
            contract("zyxab,zyxbc->zyxac", U[1], backend.roll(colvec, -1, 1)) +
            contract("zyxab,zyxbc->zyxac", U[2], backend.roll(colvec, -1, 0)) +
            backend.roll(contract("zyxab,zyxbc->zyxac", U_dag[0], colvec), 1, 2) +
            backend.roll(contract("zyxab,zyxbc->zyxac", U_dag[1], colvec), 1, 1) +
            backend.roll(contract("zyxab,zyxbc->zyxac", U_dag[2], colvec), 1, 0)
        )
    ).reshape(Lz * Ly * Lx * Nc, -1)


class EigenvectorGenerator:
    def __init__(self, latt_size, gauge_field, Ne, tol) -> None:
        Lx, Ly, Lz, Lt = latt_size
        self.latt_size = latt_size
        self.gauge_field = gauge_field
        self.Ne = Ne
        self.tol = tol
        self._U = None
        self._gauge_field_path = None

    def load(self, key: str):
        self._U = self.gauge_field.load(key)[:].transpose(4, 0, 1, 2, 3, 5, 6)[:Nd - 1]
        self._gauge_field_path = self.gauge_field.load(key).file

    def porject_SU3(self):
        U = self._U
        Uinv = backend.linalg.inv(U)
        while (backend.max(
            backend.abs(U - contract("...ab->...ba", Uinv.conj()))
        ) > 1e-15) or (backend.max(backend.abs(contract("...ab,...cb", U, U.conj()) - backend.identity(Nc))) > 1e-15):
            U = 0.5 * (U + contract("...ab->...ba", Uinv.conj()))
            Uinv = backend.linalg.inv(U)
        self._U = U

    def stout_smear(self, nstep, rho):
        # from pyquda import core
        # from pyquda.utils import gauge_utils

        # gauge = gauge_utils.readIldg(self._gauge_field_path)
        # latt_size = gauge.latt_size
        # Lx, Ly, Lz, Lt = latt_size

        # core.smear(gauge.latt_size, gauge, nstep, rho)
        # self._U = gauge.lexico().reshape(Nd, Lt, Lz, Ly, Lx, Nc, Nc)

        Lx, Ly, Lz, Lt = self.latt_size

        U = self._U
        for _ in range(nstep):
            Q = backend.zeros_like(U)
            for mu in range(Nd - 1):
                for nu in range(Nd - 1):
                    if (mu != nu):
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
            Q = contract(
                "...ab,...cb->...ac",
                rho * Q,
                U.conj(),
            )
            Q = 0.5j * (contract("...ab->...ba", Q.conj()) - Q)
            contract("...aa->...a",
                     Q)[:] -= 1 / Nc * contract("...aa->...", Q).reshape(Nd - 1, Lt, Lz, Ly, Lx, 1).repeat(Nc, 5)
            c0 = 1 / 3 * contract("...aa->...", contract("...ab,...bc,...cd->...ad", Q, Q, Q)).real
            c1 = 1 / 2 * contract("...aa->...", contract("...ab,...bc->...ac", Q, Q)).real
            c0_max = 2 * (c1 / 3)**(2 / 3)
            parity = backend.where(c0 < 0)
            c0[parity] *= -1
            theta = backend.arccos(c0 / c0_max)
            u = (1 / 3 * c1)**0.5 * backend.cos(1 / 3 * theta)
            w = c1**0.5 * backend.sin(1 / 3 * theta)
            xi0 = backend.zeros_like(w)
            small = backend.where(backend.abs(w) <= 0.05)
            large = backend.where(backend.abs(w) > 0.05)
            w_small_square = w[small]**2
            w_large = w[large]
            xi0[small] = 1 - 1 / 6 * w_small_square * (1 - 1 / 20 * w_small_square * (1 - 1 / 42 * w_small_square**2))
            xi0[large] = backend.sin(w_large) / w_large
            f_denominator = 9 * u**2 - w**2
            f0 = (
                (u**2 - w**2) * backend.exp(2j * u) + backend.exp(-1j * u) *
                (8 * u**2 * backend.cos(w) + 2j * u * (3 * u**2 + w**2) * xi0)
            ) / f_denominator
            f1 = (
                2 * u * backend.exp(2j * u) - backend.exp(-1j * u) *
                (2 * u * backend.cos(w) - 1j * (3 * u**2 - w**2) * xi0)
            ) / f_denominator
            f2 = (backend.exp(2j * u) - backend.exp(-1j * u) * (backend.cos(w) + 3j * u * xi0)) / f_denominator
            f0[parity] = f0[parity].conj()
            f1[parity] = -f1[parity].conj()
            f2[parity] = f2[parity].conj()
            f0 = contract("...,ab->...ab", f0, backend.identity(Nc))
            f1 = contract("...,...ab->...ab", f1, Q)
            f2 = contract("...,...ab,...bc->...ac", f2, Q, Q)
            U = contract("...ab,...bc->...ac", f0 + f1 + f2, U)
        self._U = U

    def calc(self, t: int):
        Lx, Ly, Lz, Lt = self.latt_size

        U = backend.asarray(self._U[:Nd - 1, t].copy())
        U_dag = U.transpose(0, 1, 2, 3, 5, 4).conj()
        Laplacian = functools.partial(_Laplacian, U=U, U_dag=U_dag, latt_size=self.latt_size)
        A = linalg.LinearOperator((Lz * Ly * Lx * Nc, Lz * Ly * Lx * Nc), matvec=Laplacian, matmat=Laplacian)
        evals, evecs = linalg.eigsh(A, self.Ne, which="SA", tol=self.tol)

        # [Ne, Lz * Ly * Lx, Nc]
        return evecs.transpose(1, 0).reshape(self.Ne, Lz * Ly * Lx, Nc)
