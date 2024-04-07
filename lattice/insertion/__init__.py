from typing import Dict, List

from .gamma import (
    GammaName,
    scheme as gamma_scheme,
    group as gamma_gourp,
    parity as gamma_parity,
    charge_conjugation as gamma_charge_conjugation,
    hermiticity as gamma_hermiticity,
)
from .derivative import (
    DerivativeName,
    scheme as derivative_scheme,
    group as derivative_gourp,
    parity as derivative_parity,
    charge_conjugation as derivative_charge_conjugation,
    hermiticity as derivative_hermiticity,
)


class ProjectionName:
    A1 = "A_1"
    A2 = "A_2"
    E = "E"
    T1 = "T_1"
    T2 = "T_2"


class InsertionRowMom:
    def __init__(self, row, momentum) -> None:
        self.row = row
        self.momentum = momentum


class Operator:
    def __init__(self, name: str, insertion_rows: List[InsertionRowMom], coefficients: List[float]) -> None:
        assert len(insertion_rows) == len(
            coefficients
        ), f"Unmatched numbers of insertion rows {len(insertion_rows)} and coefficients {len(coefficients)}"
        parts = []
        for idx in range(len(insertion_rows)):
            row, momentum, coefficient = insertion_rows[idx].row, insertion_rows[idx].momentum, coefficients[idx]
            for i in range(len(row) // 2):
                parts.append(row[i * 2])
                elemental_part = []
                for derivative_coeff, derivative_idx in row[i * 2 + 1]:
                    if parts[-1] == 5 or parts[-1] == 13:
                        # gamma_3gamma_1 = -gamma(5), gamma_3gamma_1gamma_4 = -gamma(13)
                        derivative_coeff *= -1
                    elemental_part.append([coefficient * derivative_coeff, derivative_idx, momentum])
                parts.append(elemental_part)

        self.name = name
        self.parts = parts

    def __str__(self) -> str:
        ret = ""
        ret += f"============== operator {self.name}, components: ===============\n"
        for irow in range(len(self.parts) // 2):
            ret += f"   gamma idx = {str(self.parts[2*irow])}, \n"
            for iterm in self.parts[2 * irow + 1]:
                coeff, derivative_idx, momentum = iterm
                ret += f"       > coeff = {coeff}, derivative_idx = {derivative_idx}, momentum = {momentum}\n"
        ret += f"================================================================\n"
        return ret

    def set_gamma(self, i_row, gamma_idx):
        """
        Set gamma indix for irow-th InsertionRow.
        """
        self.parts[2 * i_row] = gamma_idx

    def set_derivative(self, i_row, i_term, deriv_idx):
        """
        Set derivative indix for i-th term of  i-th InsertionRow.
        """
        self.parts[2 * i_row + 1][i_term][1] = deriv_idx


class OperatorDisplacement(Operator):
    def __init__(
        self, name: str, insertion_rows: List[InsertionRowMom], coefficients: List[float], distance: int
    ) -> None:
        super().__init__(name, insertion_rows, coefficients)
        for irow in range(len(self.parts) // 2):
            for iterm, term in enumerate(self.parts[2 * irow + 1]):
                coeff, derivative_idx, momentum = term
                assert (
                    derivative_idx == 0
                ), f"displacement operator cannot define at derivative_idx = {derivative_idx}, not 0"
                self.set_derivative(i_row=irow, i_term=iterm, deriv_idx=distance)


class InsertionRow:
    def __init__(self, row, momentum_dict) -> None:
        self.row = row
        self.momentum_dict = momentum_dict

    def __call__(self, npx, npy, npz) -> InsertionRowMom:
        return InsertionRowMom(self.row, list(self.momentum_dict.values()).index(f"{npx} {npy} {npz}"))

    def __str__(self) -> str:
        from .gamma import output as gamma_str
        from .derivative import output as derivative_str

        ret = ""
        parts = self.row
        for i in range(len(parts) // 2):
            derivative_part = parts[i * 2 + 1]
            derivative_str_part = ""
            for j in range(len(derivative_part)):
                derivative_str_part += f"{derivative_str(derivative_part[j])}"
                if j != len(derivative_part) - 1:
                    derivative_str_part += " + "
            ret += f"{gamma_str(parts[i*2])} * ({derivative_str_part})"
            if i != len(parts) // 2 - 1:
                ret += " + "
        return ret


class Insertion:
    def __init__(
        self, gamma: GammaName, derivative: DerivativeName, projection: ProjectionName, momentum_dict: Dict[int, str]
    ) -> None:
        self.gamma = gamma_scheme(gamma)
        self.derivative = derivative_scheme(derivative)
        self.parity = gamma_parity(gamma) * derivative_parity(derivative)
        self.charge_conjugation = gamma_charge_conjugation(gamma) * derivative_charge_conjugation(derivative)
        self.hermiticity = gamma_hermiticity(gamma) * derivative_hermiticity(derivative)
        self.projection = [gamma_gourp(gamma), derivative_gourp(derivative), projection]
        self.momentum_dict = momentum_dict
        self.rows = []
        self.construct()

    def __getitem__(self, idx) -> InsertionRow:
        return InsertionRow(self.rows[idx], self.momentum_dict)

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
        length = {"A_1": 1, "A_2": 1, "E": 2, "T_1": 3, "T_2": 3}
        irrep_T1 = {
            "A_1": ["T_1"],
            "E": ["T_1", "T_2"],
            "T_1": ["A_1", "T_1", "T_2", "E"],
            "T_2": ["T_1", "T_2", "E", "A_2"],
        }
        if left == "A_1":
            assert right == projection, f"{left} x {right} has no irrep {projection}"
            for i in range(length[projection]):
                self.rows.append([gamma[0], derivative[i]])
        elif left == "T_1":
            assert projection in irrep_T1[right], f"{left} x {right} has no irrep {projection}"
            for i in range(length[projection]):
                if right == "A_1":
                    self.rows.append([gamma[i], derivative[0]])
                elif right == "E":
                    raise NotImplementedError(f"{left} x {right} not implemented yet")
                elif projection in ["A_1", "A_2"]:
                    self.rows.append([gamma[0], derivative[0], gamma[1], derivative[1], gamma[2], derivative[2]])
                elif projection in ["E"]:
                    if i == 0:
                        self.rows.append([gamma[0], derivative[0], gamma[1], self.multiply(-1, derivative[1])])
                    else:
                        self.rows.append(
                            [
                                gamma[0],
                                self.multiply(-1, derivative[0]),
                                gamma[1],
                                self.multiply(-1, derivative[1]),
                                gamma[2],
                                self.multiply(2, derivative[2]),
                            ]
                        )
                elif projection in ["T_1", "T_2"]:
                    j = (i + 1) % 3
                    k = (i + 2) % 3
                    if right == projection:
                        self.rows.append([gamma[j], derivative[k], gamma[k], self.multiply(-1, derivative[j])])
                    else:
                        self.rows.append([gamma[j], derivative[k], gamma[k], derivative[j]])
