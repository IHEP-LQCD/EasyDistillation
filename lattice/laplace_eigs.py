import functools
from time import perf_counter
if False:
    import numpy as np
    from scipy.sparse import linalg
else:
    import cupy as np
    from cupyx.scipy.sparse import linalg
from opt_einsum import contract

from pyquda import core
from pyquda.utils import gauge_utils
from pyquda.field import Nc, Nd


def _Amatmat(colvec, colmat, colmat_dag, latt_size):
    Lx, Ly, Lz, Lt = latt_size
    colvec = colvec.reshape(Lz, Ly, Lx, Nc, -1)
    return (
        # - for SA with evals , + for LA with (12 - evals)
        6 * colvec + (
            contract("zyxab,zyxbc->zyxac", colmat[0], np.roll(colvec, -1, 2)) +
            contract("zyxab,zyxbc->zyxac", colmat[1], np.roll(colvec, -1, 1)) +
            contract("zyxab,zyxbc->zyxac", colmat[2], np.roll(colvec, -1, 0)) +
            np.roll(contract("zyxab,zyxbc->zyxac", colmat_dag[0], colvec), 1, 2) +
            np.roll(contract("zyxab,zyxbc->zyxac", colmat_dag[1], colvec), 1, 1) +
            np.roll(contract("zyxab,zyxbc->zyxac", colmat_dag[2], colvec), 1, 0)
        )
    ).reshape(Lz * Ly * Lx * Nc, -1)


def smear_gauge(gauge_path: str, nstep: int, rho: float):
    gauge = gauge_utils.readIldg(gauge_path)
    latt_size = gauge.latt_size
    Lx, Ly, Lz, Lt = latt_size

    core.smear(gauge.latt_size, gauge, nstep, rho)
    return gauge.lexico().reshape(Nd, Lt, Lz, Ly, Lx, Nc, Nc)[:3]


def calc_laplace_eigs(colmat_all, num_evecs: int, tol: float):
    latt_size = list(colmat_all.shape[1:5])
    Lx, Ly, Lz, Lt = latt_size
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

    # [Lt, Ne, Lz * Ly * Lx, Nc]
    return evecs.transpose(0, 2, 1).reshape(Lt, Ne, Lz * Ly * Lx, Nc)
