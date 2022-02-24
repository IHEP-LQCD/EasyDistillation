from typing import List

from .gamma import GAMMA_NAME
from .gamma import scheme as gamma_scheme
from .derivative import DERIV_NAME
from .derivative import scheme as deriv_scheme


class OperatorPart:
    def __init__(self, coeff: int, gamma, deriv) -> None:
        from .derivative import Deriv

        self.coeff: int = coeff
        self.gamma: List[int] = gamma
        self.deriv: Deriv = deriv

    def normalize(self, sumsq: int):
        for part in self.deriv.parts:
            part.coeff *= self.coeff * sumsq ** -0.5
        self.coeff = 1

    def __repr__(self) -> str:
        return f"{self.gamma} {self.deriv}"


class Operator:
    def __init__(self, parts: list) -> None:
        from copy import deepcopy as cp

        sumsq = 0
        self.parts: List[OperatorPart] = []
        for part in parts:
            sumsq += part[0] ** 2
            self.parts.append(OperatorPart(part[0], part[1], cp(part[2])))
        for part in self.parts:
            part.normalize(sumsq)

    def __repr__(self) -> str:
        ret = [part for part in self.parts]
        return f"{ret}"


def only(gamma_name: GAMMA_NAME):
    from .gamma import scheme as gamma_scheme
    from .derivative import scheme as deriv_scheme

    gamma = gamma_scheme(gamma_name)
    deriv = deriv_scheme("_")
    return [
        Operator(
            [
                [1, gamma[0], deriv[0]],
            ]
        ),
        Operator(
            [
                [1, gamma[1], deriv[0]],
            ]
        ),
        Operator(
            [
                [1, gamma[2], deriv[0]],
            ]
        ),
    ]


def multiply(gamma_name: GAMMA_NAME, deriv_name: DERIV_NAME):
    gamma = gamma_scheme(gamma_name)
    deriv = deriv_scheme(deriv_name)
    return [
        Operator(
            [
                [1, gamma[0], deriv[0]],
            ]
        ),
        Operator(
            [
                [1, gamma[0], deriv[1]],
            ]
        ),
        Operator(
            [
                [1, gamma[0], deriv[2]],
            ]
        ),
    ]


def dot(gamma_name: GAMMA_NAME, deriv_name: DERIV_NAME):
    gamma = gamma_scheme(gamma_name)
    deriv = deriv_scheme(deriv_name)
    return [
        Operator(
            [
                [1, gamma[0], deriv[0]],
                [1, gamma[1], deriv[1]],
                [1, gamma[2], deriv[2]],
            ]
        )
    ]


def epsilon_ijk(gamma_name: GAMMA_NAME, deriv_name: DERIV_NAME):
    gamma = gamma_scheme(gamma_name)
    deriv = deriv_scheme(deriv_name)
    return [
        Operator(
            [
                [1, gamma[1], deriv[2]],
                [-1, gamma[2], deriv[1]],
            ]
        ),
        Operator(
            [
                [1, gamma[2], deriv[0]],
                [-1, gamma[0], deriv[2]],
            ]
        ),
        Operator(
            [
                [1, gamma[0], deriv[1]],
                [-1, gamma[1], deriv[0]],
            ]
        ),
    ]


def abs_epslion_ijk(gamma_name: GAMMA_NAME, deriv_name: DERIV_NAME):
    gamma = gamma_scheme(gamma_name)
    deriv = deriv_scheme(deriv_name)
    return [
        Operator(
            [
                [1, gamma[1], deriv[2]],
                [1, gamma[2], deriv[1]],
            ]
        ),
        Operator(
            [
                [1, gamma[2], deriv[0]],
                [1, gamma[0], deriv[2]],
            ]
        ),
        Operator(
            [
                [1, gamma[0], deriv[1]],
                [1, gamma[1], deriv[0]],
            ]
        ),
    ]


def Q_ijk(gamma_name: GAMMA_NAME, deriv_name: DERIV_NAME):
    gamma = gamma_scheme(gamma_name)
    deriv = deriv_scheme(deriv_name)
    return [
        Operator(
            [
                [1, gamma[0], deriv[0]],
                [-1, gamma[1], deriv[1]],
            ]
        ),
        Operator(
            [
                [-1, gamma[0], deriv[0]],
                [-1, gamma[1], deriv[1]],
                [2, gamma[2], deriv[2]],
            ]
        ),
    ]
