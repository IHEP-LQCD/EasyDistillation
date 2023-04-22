import functools
if True:
    import numpy as np
    from scipy.sparse import linalg
else:
    import cupy as np
    from cupyx.scipy.sparse import linalg
from opt_einsum import contract

from ..constant import Nc, Nd

from pyquda import core
from pyquda.utils import gauge_utils


def _Laplacian(colvec, U, U_dag, latt_size):
    Lx, Ly, Lz, Lt = latt_size
    colvec = colvec.reshape(Lz, Ly, Lx, Nc, -1)
    return (
        # - for SA with evals , + for LA with (12 - evals)
        6 * colvec - (
            contract("zyxab,zyxbc->zyxac", U[0], np.roll(colvec, -1, 2)) +
            contract("zyxab,zyxbc->zyxac", U[1], np.roll(colvec, -1, 1)) +
            contract("zyxab,zyxbc->zyxac", U[2], np.roll(colvec, -1, 0)) +
            np.roll(contract("zyxab,zyxbc->zyxac", U_dag[0], colvec), 1, 2) +
            np.roll(contract("zyxab,zyxbc->zyxac", U_dag[1], colvec), 1, 1) +
            np.roll(contract("zyxab,zyxbc->zyxac", U_dag[2], colvec), 1, 0)
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
        self._U = self.gauge_field.load(key)[:].transpose(4, 0, 1, 2, 3, 5, 6)
        self._gauge_field_path = self.gauge_field.load(key).file

    def stout_smear(self, nstep, rho):
        gauge = gauge_utils.readIldg(self._gauge_field_path)
        latt_size = gauge.latt_size
        Lx, Ly, Lz, Lt = latt_size

        core.smear(gauge.latt_size, gauge, nstep, rho)
        self._U = gauge.lexico().reshape(Nd, Lt, Lz, Ly, Lx, Nc, Nc)

    def calc(self, t: int):
        Lx, Ly, Lz, Lt = self.latt_size

        U = np.asarray(self._U[:Nd - 1, t].copy())
        U_dag = U.transpose(0, 1, 2, 3, 5, 4).conj()
        Laplacian = functools.partial(_Laplacian, U=U, U_dag=U_dag, latt_size=self.latt_size)
        A = linalg.LinearOperator((Lz * Ly * Lx * Nc, Lz * Ly * Lx * Nc), matvec=Laplacian, matmat=Laplacian)
        evals, evecs = linalg.eigsh(A, self.Ne, which="SA", tol=self.tol)

        # [Ne, Lz * Ly * Lx, Nc]
        return evecs.transpose(1, 0).reshape(self.Ne, Lz * Ly * Lx, Nc)
