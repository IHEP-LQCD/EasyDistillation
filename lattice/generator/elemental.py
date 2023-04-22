from math import factorial
from typing import List, Tuple

from ..constant import Nc, Nd
from ..backend import get_backend
from ..preset import GaugeField, Eigenvector
from ..insertion.phase import MomentumPhase


def comb(n, i):
    return factorial(n) // (factorial(i) * factorial(n - i))


class ElementalGenerator:
    def __init__(
        self,
        latt_size: List[int],
        gauge_field: GaugeField,
        eigenvector: Eigenvector,
        num_nabla: int = 0,
        momentum_list: List[Tuple[int]] = [(0, 0, 0)]
    ) -> None:
        from ..insertion.derivative import derivative
        backend = get_backend()
        Lx, Ly, Lz, Lt = latt_size

        self.latt_size = latt_size
        self.gauge_field = gauge_field
        self.eigenvector = eigenvector
        self.num_derivative = (3**(num_nabla + 1) - 1) // 2
        self.derivative_list = [derivative(n) for n in range(self.num_derivative)]
        self.num_momentum = len(momentum_list)
        self.momentum_list = momentum_list
        Ne = eigenvector.Ne
        self.Ne = eigenvector.Ne
        self._U = backend.zeros((Nd, Lz * Ly * Lx, Nc, Nc), "<c16")
        self._V = backend.zeros((Ne, Lz * Ly * Lx, Nc), "<c8")
        self._VPV = backend.zeros((self.num_derivative, self.num_momentum, Ne, Ne), "<c16")
        self._gauge_field_data = None
        self._eigenvector_data = None
        self._momentum_phase = MomentumPhase(Lx, Ly, Lz)

    def nD(self, V, U, deriv):
        from opt_einsum import contract
        backend = get_backend()
        Lx, Ly, Lz, Lt = self.latt_size

        Ne = self.Ne
        for d in deriv:
            Vf = backend.roll(V.reshape(Ne, Lz, Ly, Lx, Nc), -1, 3 - d).reshape(Ne, -1, Nc)
            UVf = contract("xab,exb->exa", U[d], Vf)
            UdV = contract("xba,exb->exa", U[d].conj(), V)
            UbdVb = backend.roll(UdV.reshape(Ne, Lz, Ly, Lx, Nc), 1, 3 - d).reshape(Ne, -1, Nc)
            V = UVf - UbdVb
        return V

    def load(self, key: str):
        self._gauge_field_data = self.gauge_field.load(key)
        self._eigenvector_data = self.eigenvector.load(key)

    def calc(self, t: int):
        from opt_einsum import contract

        gauge_field = self._gauge_field_data
        eigenvector = self._eigenvector_data
        momentum_phase = self._momentum_phase
        U = self._U
        V = self._V
        VPV = self._VPV

        if self.derivative_list != [()]:
            for d in range(U.shape[0]):
                U[d] = gauge_field[t, :, d]
        for e in range(V.shape[0]):
            V[e] = eigenvector[t, e]
        for derivative_idx, derivative in enumerate(self.derivative_list):
            VPV[derivative_idx] = 0
            for num_nabla_right in range(len(derivative) + 1):
                coeff = (-1)**num_nabla_right * comb(len(derivative), num_nabla_right)
                right = self.nD(V, U, derivative[:num_nabla_right])
                left = self.nD(V, U, derivative[num_nabla_right:][::-1])
                for momentum_idx, momentum in enumerate(self.momentum_list):
                    VPV[derivative_idx, momentum_idx] += contract(
                        "x,exc,fxc->ef",
                        coeff * momentum_phase.get(momentum).reshape(-1), left.conj(), right
                    )
        return VPV
