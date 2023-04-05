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


class Operator:
    def __init__(self, parts, momentum) -> None:
        self.parts = parts
        self.momentum = momentum


class Insertion:
    def __init__(self, gamma, derivative, projection) -> None:
        self.gamma = gamma_scheme(gamma)
        self.derivative = derivative_scheme(derivative)
        self.parity = gamma_parity(gamma) * derivative_parity(derivative)
        self.charge_conjugation = gamma_charge_conjugation(gamma) * derivative_charge_conjugation(derivative)
        self.hermiticity = gamma_hermiticity(gamma) * derivative_hermiticity(derivative)
        self.projection = [gamma_gourp(gamma), derivative_gourp(derivative), projection]
        self.parts = []
        self.construct()

    def __call__(self, npx, npy, npz) -> Operator:
        return Operator(self.parts, f"{npx} {npy} {npz}")

    def __str__(self) -> str:
        from .gamma import gamma_str
        from .derivative import derivative_str
        ret = ""
        for part in self.parts:
            for i in range(len(part) // 2):
                derivative_str_part = ""
                for j in range(len(part[i * 2])):
                    derivative_str_part += F"{derivative_str(part[i*2][j])}"
                    if j != len(part[i * 2]) - 1:
                        derivative_str_part += " + "
                ret += F"{gamma_str(part[i*2+1])} * ({derivative_str_part})"
                if i != len(part) // 2 - 1:
                    ret += " + "
                else:
                    ret += "\n"
        return ret

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
                self.parts.append([derivative[i], gamma[0]])
        elif left == "T1":
            assert projection in irrep_T1[right], f"{left} x {right} has no irrep {projection}"
            for i in range(length[projection]):
                if right == "A1":
                    self.parts.append([derivative[0], gamma[i]])
                elif right == "E":
                    raise NotImplementedError(f"{left} x {right} not implemented yet")
                elif projection in ["A1", "A2"]:
                    self.parts.append([derivative[0], gamma[0], derivative[1], gamma[1], derivative[2], gamma[2]])
                elif projection in ["E"]:
                    if i == 0:
                        self.parts.append([derivative[0], gamma[0], self.multiply(-1, derivative[1]), gamma[1]])
                    else:
                        self.parts.append(
                            [
                                self.multiply(-1, derivative[0]), gamma[0],
                                self.multiply(-1, derivative[1]), gamma[1],
                                self.multiply(2, derivative[2]), gamma[2]
                            ]
                        )
                elif projection in ["T1", "T2"]:
                    j = (i + 1) % 3
                    k = (i + 2) % 3
                    if right == projection:
                        self.parts.append([derivative[k], gamma[j], self.multiply(-1, derivative[j]), gamma[k]])
                    else:
                        self.parts.append([derivative[k], gamma[j], derivative[j], gamma[k]])


# class OperatorPart:
#     def __init__(self, coeff: int, gamma, deriv) -> None:
#         from .derivative import Deriv

#         self.coeff: int = coeff
#         self.gamma: List[int] = gamma
#         self.deriv: Deriv = deriv

#     def normalize(self, sumsq: int):
#         for part in self.deriv.parts:
#             part.coeff *= self.coeff * sumsq**-0.5
#         self.coeff = 1

#     def __repr__(self) -> str:
#         return f"gamma({self.gamma}) * {self.deriv}"

# class Operator:
#     def __init__(self, parts: list, hermition: int) -> None:
#         from copy import deepcopy as cp

#         sumsq = 0
#         self.parts: List[OperatorPart] = []
#         for part in parts:
#             sumsq += part[0]**2
#             self.parts.append(OperatorPart(part[0], part[1], cp(part[2])))
#         for part in self.parts:
#             part.normalize(sumsq)
#         self.hermition = hermition

#     def __repr__(self) -> str:
#         ret = [part for part in self.parts]
#         return f"{ret}"

# def only_gamma(gamma_name: GAMMA_NAME):
#     gamma = gamma_scheme(gamma_name)
#     deriv = deriv_scheme("")
#     hermition = gamma_hermition(gamma_name) * deriv_hermition("")
#     return [Operator([
#         [1, gamma[i], deriv[0]],
#     ], hermition) for i in range(len(gamma))]

# def multiply(gamma_name: GAMMA_NAME, deriv_name: DERIV_NAME):
#     gamma = gamma_scheme(gamma_name)
#     deriv = deriv_scheme(deriv_name)
#     hermition = gamma_hermition(gamma_name) * deriv_hermition(deriv_name)
#     assert len(gamma) == 1 and len(deriv) == 3
#     return [
#         Operator([
#             [1, gamma[0], deriv[0]],
#         ], hermition),
#         Operator([
#             [1, gamma[0], deriv[1]],
#         ], hermition),
#         Operator([
#             [1, gamma[0], deriv[2]],
#         ], hermition),
#     ]

# def dot(gamma_name: GAMMA_NAME, deriv_name: DERIV_NAME):
#     gamma = gamma_scheme(gamma_name)
#     deriv = deriv_scheme(deriv_name)
#     hermition = gamma_hermition(gamma_name) * deriv_hermition(deriv_name)
#     assert len(gamma) == 3 and len(deriv) == 3
#     return [Operator([
#         [1, gamma[0], deriv[0]],
#         [1, gamma[1], deriv[1]],
#         [1, gamma[2], deriv[2]],
#     ], hermition)]

# def epsilon_ijk(gamma_name: GAMMA_NAME, deriv_name: DERIV_NAME):
#     gamma = gamma_scheme(gamma_name)
#     deriv = deriv_scheme(deriv_name)
#     hermition = gamma_hermition(gamma_name) * deriv_hermition(deriv_name)
#     assert len(gamma) == 3 and len(deriv) == 3
#     return [
#         Operator([
#             [1, gamma[1], deriv[2]],
#             [-1, gamma[2], deriv[1]],
#         ], hermition),
#         Operator([
#             [1, gamma[2], deriv[0]],
#             [-1, gamma[0], deriv[2]],
#         ], hermition),
#         Operator([
#             [1, gamma[0], deriv[1]],
#             [-1, gamma[1], deriv[0]],
#         ], hermition),
#     ]

# def abs_epslion_ijk(gamma_name: GAMMA_NAME, deriv_name: DERIV_NAME):
#     gamma = gamma_scheme(gamma_name)
#     deriv = deriv_scheme(deriv_name)
#     hermition = gamma_hermition(gamma_name) * deriv_hermition(deriv_name)
#     assert len(gamma) == 3 and len(deriv) == 3
#     return [
#         Operator([
#             [1, gamma[1], deriv[2]],
#             [1, gamma[2], deriv[1]],
#         ], hermition),
#         Operator([
#             [1, gamma[2], deriv[0]],
#             [1, gamma[0], deriv[2]],
#         ], hermition),
#         Operator([
#             [1, gamma[0], deriv[1]],
#             [1, gamma[1], deriv[0]],
#         ], hermition),
#     ]

# def Q_ijk(gamma_name: GAMMA_NAME, deriv_name: DERIV_NAME):
#     gamma = gamma_scheme(gamma_name)
#     deriv = deriv_scheme(deriv_name)
#     hermition = gamma_hermition(gamma_name) * deriv_hermition(deriv_name)
#     assert len(gamma) == 3 and len(deriv) == 3
#     return [
#         Operator([
#             [1, gamma[0], deriv[0]],
#             [-1, gamma[1], deriv[1]],
#         ], hermition),
#         Operator([
#             [-1, gamma[0], deriv[0]],
#             [-1, gamma[1], deriv[1]],
#             [2, gamma[2], deriv[2]],
#         ], hermition),
#     ]
