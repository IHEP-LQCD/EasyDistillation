from typing import List, Literal

from opt_einsum import contract

from ..constant import Nc, Ns, Nd
from ..backend import set_backend, get_backend, check_QUDA
from ..preset import GaugeField, Eigenvector


class PerambulatorGenerator:
    """
     Generate perambulators in distillation,
        based on PyQUDA + QUDA for GPU-accelerated computations.

    Parameters:
    -----------
    latt_size : List[int]
        dimensions of the lattice, order as [Lx, Ly, Lz, Lt].
    gauge_field : GaugeField.
    eigenvector : Eigenvector.
    mass : float
        The mass parameter for the Dirac operator.
    tol : float
        The Dirac operator tol.
    maxiter : int
        The maximum number of iterations of solver.
    xi_0 : float, optional
        The anisotropy, defaults to 1.0.
    nu : float, optional.
    clover_coeff_t : float, optional
        The temporal clover coefficient, defaults to 0.0.
    clover_coeff_r : float, optional
        The spatial clover coefficient, defaults to 1.0.
    t_boundary : Literal[1, -1], optional
        The temporal boundary condition, defaults to 1 (periodic).
    multigrid : List[List[int]], optional
        The multigrid levels for the solver, defaults to None.
    contract_prec : str, optional
        The precision for the contraction operations, defaults to '<c16'.
    usedNe : int, optianal
        The used eigenvectors number, defaults to None, usedNe is eigenvector.Ne .
    MRHS : bool, optional:
        Use MRHS methods to solve perambulators, defaults to False.
        This option requires more device memory.

    Notes:
    ------
    - This class requires PyQUDA + QUDA for GPU-accelerated computations.
    """

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
        contract_prec: str = "<c16",
        usedNe: int = None,
        MRHS: bool = False,
    ) -> None:
        if not check_QUDA():
            raise ImportError("Please install PyQuda to generate the perambulator or check MPI_init again.")
        from pyquda_utils import core

        self.latt_info = core.LatticeInfo(latt_size=latt_size, t_boundary=t_boundary, anisotropy=xi_0 / nu)
        self.contract_prec = contract_prec

        backend = get_backend()
        assert backend.__name__ == "cupy", "PyQuda only support cupy as the ndarray implementation"
        Lx, Ly, Lz, Lt = self.latt_info.size
        if usedNe is None:
            usedNe = eigenvector.Ne
        elif eigenvector.Ne != usedNe:
            print(f"Warning: used Ne = {usedNe}, data maximum Ne = {eigenvector.Ne}")
        self.usedNe = usedNe
        Ne = usedNe

        self.gauge_field = gauge_field
        self.gauge_field_smear = None
        self.gauge_field_new = None
        self.eigenvector = eigenvector
        self.MRHS = MRHS
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
        self._SV = backend.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc), self.contract_prec)
        self._VSV = backend.zeros((Lt, Ns, Ns, Ne, Ne), self.contract_prec)

    def load(self, key: str):
        import numpy as np
        from pyquda_utils import core
        from pyquda_utils import io

        backend = get_backend()
        Lx, Ly, Lz, Lt = self.latt_info.size
        gx, gy, gz, gt = self.latt_info.grid_coord
        Ne = self.usedNe
        self.gauge_field_smear = io.readQIOGauge(self.gauge_field.load(key).file)
        self.gauge_field_new = True

        eigenvector_data = self.eigenvector.load(key)
        eigenvector_data_dagger = np.zeros((Ne, Lt, Lz, Ly, Lx, Nc), self.contract_prec)
        # read data into host memory
        # save V^\dag here to save device memory
        set_backend("numpy")
        for e in range(Ne):
            for t in range(Lt):
                eigenvector_data_dagger[e, t] = eigenvector_data[
                    gt * Lt + t, e, gz * Lz : (gz + 1) * Lz, gy * Ly : (gy + 1) * Ly, gx * Lx : (gx + 1) * Lx
                ].conj()
        set_backend(backend)
        # set eigenvector_data_cb2 on device mem
        self._eigenvector_data_dagger = backend.asarray(core.cb2(eigenvector_data_dagger, [1, 2, 3, 4]))

    def _stout_smear_quda(self, nstep: int, rho: float, dir_ignore: int):
        gauge = self.gauge_field_smear
        if self.gauge_field_smear is None:
            raise ValueError("Gauge not loaded, please use .load() before .stout_smear().")

        gauge.smearSTOUT(nstep, rho, dir_ignore)
        self.gauge_field_smear = gauge

    def stout_smear(self, nstep: int, rho: float, dir_ignore: int = 3):
        backend = get_backend()
        if backend.__name__ == "numpy":
            raise NotImplementedError("Ndarray stout smear not implement in PerambulatorGenerator.")
        elif backend.__name__ == "cupy":
            # __init__() has check_QUDA() before !
            self._stout_smear_quda(nstep, rho, dir_ignore)

    def calc(self, t: int):
        import cupy as cp

        backend = get_backend()
        from pyquda_utils.core import LatticeFermion, MultiLatticeFermion

        if self.gauge_field_new:
            self.dirac.loadGauge(self.gauge_field_smear)  # loadGauge after
            self.gauge_field_new = False

        latt_info = self.latt_info
        Lx, Ly, Lz, Lt = latt_info.size
        Vol = Lx * Ly * Lz * Lt
        Ne = self.usedNe
        eigenvector_dagger = self._eigenvector_data_dagger
        dirac = self.dirac
        gx, gy, gz, gt = self.latt_info.grid_coord

        SV = self._SV
        VSV = self._VSV

        from time import perf_counter

        for eigen in range(Ne):
            cp.cuda.runtime.deviceSynchronize()
            s = perf_counter()
            if self.MRHS:
                print("Warning: use MRHS.")
                V_MRHS = MultiLatticeFermion(latt_info, Ns)
                for spin in range(Ns):
                    data = V_MRHS[spin].data.reshape(2, Lt, Lz, Ly, Lx // 2, Ns, Nc)
                    if gt * Lt <= t and (gt + 1) * Lt > t:
                        data[:, t % Lt, :, :, :, spin, :] = backend.asarray(
                            eigenvector_dagger[eigen, :, t % Lt, :, :, :, :].conj()
                        )
                SV_MRHS = dirac.invertMultiSrc(V_MRHS)
                for spin in range(Ns):
                    SV.reshape(Vol, Ns, Ns, Nc)[:, :, spin, :] = SV_MRHS[spin].data.reshape(Vol, Ns, Nc)
            else:
                for spin in range(Ns):
                    V = LatticeFermion(latt_info)  # V.data is double prec.
                    data = V.data.reshape(2, Lt, Lz, Ly, Lx // 2, Ns, Nc)
                    if gt * Lt <= t and (gt + 1) * Lt > t:
                        data[:, t % Lt, :, :, :, spin, :] = backend.asarray(
                            eigenvector_dagger[eigen, :, t % Lt, :, :, :, :].conj()
                        )  # [Ne, etzyx, Nc]
                    SV.reshape(Vol, Ns, Ns, Nc)[:, :, spin, :] = dirac.invert(V).data.reshape(Vol, Ns, Nc)  # .get()
            cp.cuda.runtime.deviceSynchronize()
            invert_time = perf_counter() - s

            cp.cuda.runtime.deviceSynchronize()
            s = perf_counter()
            VSV[:, :, :, :, eigen] = contract(
                "ketzyxa,etzyxija->tijk", backend.asarray(eigenvector_dagger), backend.asarray(SV), optimize=True
            )
            cp.cuda.runtime.deviceSynchronize()
            contraction_time = perf_counter() - s

            # print for check device mem
            free, total = cp.cuda.runtime.memGetInfo()
            print(
                f"Ne = {eigen}:  inv t = {invert_time:.4f} sec, contraction t = {contraction_time:.4f} sec, device mem: {(total - free) / 1024**3} GB, free:{free / 1024**3} GB."
            )
        return VSV
