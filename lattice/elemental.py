from math import factorial
from typing import List, Tuple

from .backend import get_backend
from .preset import GaugeField, EigenVector
from .insertion.phase import MomentumPhase

Nd = 4
Nc = 3


def comb(n, i):
    return factorial(n) // (factorial(i) * factorial(n - i))


class ElementalGenerator:
    def __init__(
        self,
        latt_size: List[int],
        gauge_field: GaugeField,
        eigen_vector: EigenVector,
        num_nabla: int = 0,
        momentum_list: List[Tuple[int]] = [(0, 0, 0)]
    ) -> None:
        from .insertion.derivative import derivative
        import numpy as numpy_ori
        numpy = get_backend()
        Lx, Ly, Lz, Lt = latt_size

        self.latt_size = latt_size
        self.gauge_field = gauge_field
        self.eigen_vector = eigen_vector
        self.momentum_phase = MomentumPhase(Lx, Ly, Lz)
        num_derivative = (3**(num_nabla + 1) - 1) // 2
        self.derivative_list = [derivative(n) for n in range(num_derivative)]
        self.momentum_list = momentum_list
        Ne = eigen_vector.Ne
        self.Ne = eigen_vector.Ne
        self.U = numpy.zeros((Nd, Lz * Ly * Lx, Nc, Nc), "<c16")
        self.V = numpy.zeros((Ne, Lz * Ly * Lx, Nc), "<c8")
        self.VPV = numpy.zeros((num_derivative, len(momentum_list), Ne, Ne), "<c16")
        self.elemental = numpy_ori.zeros((Lt, num_derivative, len(momentum_list), Ne, Ne), "<c16")

    def nD(self, V, U, deriv):
        from opt_einsum import contract
        numpy = get_backend()
        Lx, Ly, Lz, Lt = self.latt_size

        Ne = self.Ne
        for d in deriv:
            Vf = numpy.roll(V.reshape(Ne, Lz, Ly, Lx, Nc), -1, 3 - d).reshape(Ne, -1, Nc)
            UVf = contract("xab,exb->exa", U[d], Vf)
            UdV = contract("xba,exb->exa", U[d].conj(), V)
            UbdVb = numpy.roll(UdV.reshape(Ne, Lz, Ly, Lx, Nc), 1, 3 - d).reshape(Ne, -1, Nc)
            V = UVf - UbdVb
        return V

    def load(self, key: str):
        from opt_einsum import contract
        Lx, Ly, Lz, Lt = self.latt_size

        gauge_field = self.gauge_field.load(key)
        eigen_vector = self.eigen_vector.load(key)
        momentum_phase = self.momentum_phase
        # assert self.latt_size == gauge_field.latt_size and self.latt_size == eigen_vector.latt_size
        U = self.U
        V = self.V
        VPV = self.VPV
        for t in range(Lt):
            print(t)
            if self.derivative_list != [()]:
                for d in range(U.shape[0]):
                    U[d] = gauge_field[t, :, d]
            for e in range(V.shape[0]):
                V[e] = eigen_vector[t, e]
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
            self.elemental[t] = VPV.get()
