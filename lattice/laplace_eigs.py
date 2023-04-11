import functools
from time import perf_counter
if False:
    import numpy as np
    from scipy.sparse import linalg
else:
    import cupy as np
    from cupyx.scipy.sparse import linalg

from pyquda import core
from pyquda.utils import gauge_utils
from pyquda.field import Nc, Nd


def _Amatmat(colvec, colmat, colmat_dag, latt_size):
    Lx, Ly, Lz, Lt = latt_size
    colvec = colvec.reshape(Lz, Ly, Lx, Nc, -1)
    return (
        # - for SA with evals , + for LA with (12 - evals)
        6 * colvec + (
            np.einsum("zyxab,zyxbc->zyxac", colmat[0], np.roll(colvec, -1, 2)) +
            np.einsum("zyxab,zyxbc->zyxac", colmat[1], np.roll(colvec, -1, 1)) +
            np.einsum("zyxab,zyxbc->zyxac", colmat[2], np.roll(colvec, -1, 0)) +
            np.roll(np.einsum("zyxab,zyxbc->zyxac", colmat_dag[0], colvec), 1, 2) +
            np.roll(np.einsum("zyxab,zyxbc->zyxac", colmat_dag[1], colvec), 1, 1) +
            np.roll(np.einsum("zyxab,zyxbc->zyxac", colmat_dag[2], colvec), 1, 0)
        )
    ).reshape(Lz * Ly * Lx * Nc, -1)


def calcLaplaceEigs(gauge_path: str, evecs_path: str, nstep: int, rho: float, num_evecs: int, tol: float):
    gauge = gauge_utils.readIldg(gauge_path)
    latt_size = gauge.latt_size
    Lx, Ly, Lz, Lt = latt_size

    core.smear(gauge.latt_size, gauge, nstep, rho)
    colmat_all = gauge.lexico().reshape(Nd, Lt, Lz, Ly, Lx, Nc, Nc)[:3]

    Ne = num_evecs
    evecs = np.zeros((Lt, Lz * Ly * Lx * Nc, Ne), "<c16")
    for t in range(Lt):
        s = perf_counter()
        colmat = np.asarray(colmat_all[:, t].copy())
        colmat_dag = colmat.transpose(0, 1, 2, 3, 5, 4).conj()
        Amatmat = functools.partial(_Amatmat, colmat=colmat, colmat_dag=colmat_dag, latt_size=latt_size)
        A = linalg.LinearOperator((Lz * Ly * Lx * Nc, Lz * Ly * Lx * Nc), matvec=Amatmat, matmat=Amatmat)
        evals_t, evecs_t = linalg.eigsh(A, Ne, which="LA", tol=tol)
        evecs[t] = evecs_t
        print(FR"EASYDISTILLATION: {perf_counter()-s:.3f}sec to solve the lowest {Ne} eigensystem at t={t}.")

    # [Ne, Lt, Lz * Ly * Lx * Nc]
    np.save(evecs_path, evecs.transpose(2, 0, 1).reshape(Ne, Lt, Lz * Ly * Lx, Nc))
