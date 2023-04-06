from typing import Dict

from .gamma import (
    GAMMA_NAME,
    scheme as gamma_scheme,
    group as gamma_gourp,
    parity as gamma_parity,
    charge_conjugation as gamma_charge_conjugation,
    hermiticity as gamma_hermiticity,
)
from .derivative import (
    DERIVATIVE_NAME,
    scheme as derivative_scheme,
    group as derivative_gourp,
    parity as derivative_parity,
    charge_conjugation as derivative_charge_conjugation,
    hermiticity as derivative_hermiticity,
)


class PROJECTION_NAME:
    A1 = "A1"
    A2 = "A2"
    E = "E"
    T1 = "T1"
    T2 = "T2"


class Operator:
    def __init__(self, parts, momentum) -> None:
        self.parts = parts
        self.momentum = momentum


class InsertionRow:
    def __init__(self, parts, momenta) -> None:
        self.parts = parts
        self.momenta = momenta

    def __call__(self, npx, npy, npz) -> Operator:
        return Operator(self.parts, [list(self.momenta.values()).index(f"{npx} {npy} {npz}")])

    def __str__(self) -> str:
        from .gamma import gamma_str
        from .derivative import derivative_str
        ret = ""
        parts = self.parts
        for i in range(len(parts) // 2):
            derivative_part = parts[i * 2 + 1]
            derivative_str_part = ""
            for j in range(len(derivative_part)):
                derivative_str_part += F"{derivative_str(derivative_part[j])}"
                if j != len(derivative_part) - 1:
                    derivative_str_part += " + "
            ret += F"{gamma_str(parts[i*2])} * ({derivative_str_part})"
            if i != len(parts) // 2 - 1:
                ret += " + "
        return ret


class Insertion:
    def __init__(
        self, gamma: GAMMA_NAME, derivative: DERIVATIVE_NAME, projection: PROJECTION_NAME, momenta: Dict[int, str]
    ) -> None:
        self.gamma = gamma_scheme(gamma)
        self.derivative = derivative_scheme(derivative)
        self.parity = gamma_parity(gamma) * derivative_parity(derivative)
        self.charge_conjugation = gamma_charge_conjugation(gamma) * derivative_charge_conjugation(derivative)
        self.hermiticity = gamma_hermiticity(gamma) * derivative_hermiticity(derivative)
        self.projection = [gamma_gourp(gamma), derivative_gourp(derivative), projection]
        self.momenta = momenta
        self.rows = []
        self.construct()

    def __getitem__(self, idx) -> InsertionRow:
        return InsertionRow(self.rows[idx], self.momenta)

    def __str__(self) -> str:
        ret = []
        for i in range(len(self.rows)):
            ret.append(str(self[i]))
        return str(ret)

    def multiply(self, coeff, derivative):
        ret = []
        for i in range(len(derivative)):
            ret.append([coeff * derivative[i][0], *derivative[i][1:]])
        return ret

    def construct(self):
        gamma = self.gamma
        derivative = self.derivative
        left, right, projection = self.projection
        length = {"A1": 1, "A2": 1, "E": 2, "T1": 3, "T2": 3}
        irrep_T1 = {"A1": ["T1"], "E": ["T1", "T2"], "T1": ["A1", "T1", "T2", "E"], "T2": ["T1", "T2", "E", "A2"]}
        if left == "A1":
            assert right == projection, F"{left} x {right} has no irrep {projection}"
            for i in range(length[projection]):
                self.rows.append([gamma[0], derivative[i]])
        elif left == "T1":
            assert projection in irrep_T1[right], f"{left} x {right} has no irrep {projection}"
            for i in range(length[projection]):
                if right == "A1":
                    self.rows.append([gamma[i], derivative[0]])
                elif right == "E":
                    raise NotImplementedError(f"{left} x {right} not implemented yet")
                elif projection in ["A1", "A2"]:
                    self.rows.append([gamma[0], derivative[0], gamma[1], derivative[1], gamma[2], derivative[2]])
                elif projection in ["E"]:
                    if i == 0:
                        self.rows.append([gamma[0], derivative[0], gamma[1], self.multiply(-1, derivative[1])])
                    else:
                        self.rows.append(
                            [
                                gamma[0],
                                self.multiply(-1, derivative[0]), gamma[1],
                                self.multiply(-1, derivative[1]), gamma[2],
                                self.multiply(2, derivative[2])
                            ]
                        )
                elif projection in ["T1", "T2"]:
                    j = (i + 1) % 3
                    k = (i + 2) % 3
                    if right == projection:
                        self.rows.append([gamma[j], derivative[k], gamma[k], self.multiply(-1, derivative[j])])
                    else:
                        self.rows.append([gamma[j], derivative[k], gamma[k], derivative[j]])
