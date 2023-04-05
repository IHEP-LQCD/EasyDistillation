import functools
import os
from time import perf_counter
# if use cpu
import numpy as np
from scipy.sparse import linalg

# import cupy as np
# from cupyx.scipy.sparse import linalg

from pyquda import core, mpi
from pyquda.utils import gauge_utils
from pyquda.field import Nc, Nd
from lattice.dispatch import Dispatch

os.environ["QUDA_RESOURCE_PATH"] = ".cache"
mpi.init()


def _Amatmat(colvec, colmat, colmat_dag, latt_size):
    Lx, Ly, Lz, Lt = latt_size
    colvec = colvec.reshape(Lz, Ly, Lx, Nc, -1)
    return (
        # - for SA with evals , + for LA with (12 - evals)
        6 * colvec - (
            np.einsum("zyxab,zyxbc->zyxac", colmat[0], np.roll(colvec, -1, 2)) +
            np.einsum("zyxab,zyxbc->zyxac", colmat[1], np.roll(colvec, -1, 1)) +
            np.einsum("zyxab,zyxbc->zyxac", colmat[2], np.roll(colvec, -1, 0)) +
            np.roll(np.einsum("zyxab,zyxbc->zyxac", colmat_dag[0], colvec), 1, 2) +
            np.roll(np.einsum("zyxab,zyxbc->zyxac", colmat_dag[1], colvec), 1, 1) +
            np.roll(np.einsum("zyxab,zyxbc->zyxac", colmat_dag[2], colvec), 1, 0)
        )
    ).reshape(Lz * Ly * Lx * Nc, -1)


def laplace_eigs(gaugePath: str, eigvecPath: str, nstep: int, rho: float, num_evecs: int, tol: float):
    gauge = gauge_utils.readIldg(gaugePath)
    latt_size = gauge.latt_size
    Lx, Ly, Lz, Lt = latt_size

    core.smear(gauge.latt_size, gauge, nstep, rho)
    colmat_all = gauge.lexico().reshape(Nd, Lt, Lz, Ly, Lx, Nc, Nc)[:3]

    Ne = num_evecs
    V = np.zeros((Lt, Lz * Ly * Lx * Nc, Ne), "<c16")
    for t in range(Lt):
        s = perf_counter()
        colmat = np.asarray(colmat_all[:, t].copy())
        colmat_dag = colmat.transpose(0, 1, 2, 3, 5, 4).conj()
        Amatmat = functools.partial(_Amatmat, colmat=colmat, colmat_dag=colmat_dag, latt_size=latt_size)
        A = linalg.LinearOperator((Lz * Ly * Lx * Nc, Lz * Ly * Lx * Nc), matvec=Amatmat, matmat=Amatmat)
        evals, evecs = linalg.eigsh(A, Ne, tol=tol, which="SA")
        V[t] = evecs
        print(FR"EASYDISTILLATION: {perf_counter()-s:.3f}sec to solve the lowest {Ne} eigensystem at t={t}.")

    # [Ne, Lt, Lz * Ly * Lx * Nc]
    np.save(eigvecPath, V.transpose(2, 0, 1))


key = 0
lightkey = [
    "-0.05766",
    "-0.05862",
    "-0.05945",
    "-0.06016",
][key]

if __name__ == "__main__":
    print("Start")
    dispatcher = Dispatch("cfglist.txt", suffix=f"2.8-evecs-{key}")
    # genEvecs("3500")
    for cfg in dispatcher:
        print(cfg)

        prefix = F"/dg_hpc/LQCD/DATA/clqcd_nf2_clov_L16_T128_b2.0_ml{lightkey}_sn2_srho0.12_gg5.65_gf5.2_usg0.780268_usf0.949104/00.cfgs/clqcd_nf2_clov_L16_T128_b2.0_xi5_ml{lightkey}_cfg_"
        suffix = ".lime"
        out_prefix = F"clqcd_nf2_clov_L16_T128_b2.0_xi5_ml{lightkey}_cfg_"
        out_suffix = ".lime.npy"
        laplace_eigs(F"{prefix}{cfg}{suffix}", F"{out_prefix}{cfg}{out_suffix}", 10, 0.12, 10, 1e-7)
