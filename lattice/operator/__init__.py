from typing import List

from .gamma import GAMMA_NAME, scheme as gamma_scheme, hermition as gamma_hermition
from .derivative import DERIV_NAME, scheme as deriv_scheme, hermition as deriv_hermition


class OperatorPart:
    def __init__(self, coeff: int, gamma, deriv) -> None:
        from .derivative import Deriv

        self.coeff: int = coeff
        self.gamma: List[int] = gamma
        self.deriv: Deriv = deriv

    def normalize(self, sumsq: int):
        for part in self.deriv.parts:
            part.coeff *= self.coeff * sumsq**-0.5
        self.coeff = 1

    def __repr__(self) -> str:
        return f"gamma({self.gamma}) * {self.deriv}"


class Operator:
    def __init__(self, parts: list, hermition: int) -> None:
        from copy import deepcopy as cp

        sumsq = 0
        self.parts: List[OperatorPart] = []
        for part in parts:
            sumsq += part[0]**2
            self.parts.append(OperatorPart(part[0], part[1], cp(part[2])))
        for part in self.parts:
            part.normalize(sumsq)
        self.hermition = hermition

    def __repr__(self) -> str:
        ret = [part for part in self.parts]
        return f"{ret}"


def only_gamma(gamma_name: GAMMA_NAME):
    gamma = gamma_scheme(gamma_name)
    deriv = deriv_scheme("")
    hermition = gamma_hermition(gamma_name) * deriv_hermition("")
    return [Operator([
        [1, gamma[i], deriv[0]],
    ], hermition) for i in range(len(gamma))]


def multiply(gamma_name: GAMMA_NAME, deriv_name: DERIV_NAME):
    gamma = gamma_scheme(gamma_name)
    deriv = deriv_scheme(deriv_name)
    hermition = gamma_hermition(gamma_name) * deriv_hermition(deriv_name)
    assert len(gamma) == 1 and len(deriv) == 3
    return [
        Operator([
            [1, gamma[0], deriv[0]],
        ], hermition),
        Operator([
            [1, gamma[0], deriv[1]],
        ], hermition),
        Operator([
            [1, gamma[0], deriv[2]],
        ], hermition),
    ]


def dot(gamma_name: GAMMA_NAME, deriv_name: DERIV_NAME):
    gamma = gamma_scheme(gamma_name)
    deriv = deriv_scheme(deriv_name)
    hermition = gamma_hermition(gamma_name) * deriv_hermition(deriv_name)
    assert len(gamma) == 3 and len(deriv) == 3
    return [Operator([
        [1, gamma[0], deriv[0]],
        [1, gamma[1], deriv[1]],
        [1, gamma[2], deriv[2]],
    ], hermition)]


def epsilon_ijk(gamma_name: GAMMA_NAME, deriv_name: DERIV_NAME):
    gamma = gamma_scheme(gamma_name)
    deriv = deriv_scheme(deriv_name)
    hermition = gamma_hermition(gamma_name) * deriv_hermition(deriv_name)
    assert len(gamma) == 3 and len(deriv) == 3
    return [
        Operator([
            [1, gamma[1], deriv[2]],
            [-1, gamma[2], deriv[1]],
        ], hermition),
        Operator([
            [1, gamma[2], deriv[0]],
            [-1, gamma[0], deriv[2]],
        ], hermition),
        Operator([
            [1, gamma[0], deriv[1]],
            [-1, gamma[1], deriv[0]],
        ], hermition),
    ]


def abs_epslion_ijk(gamma_name: GAMMA_NAME, deriv_name: DERIV_NAME):
    gamma = gamma_scheme(gamma_name)
    deriv = deriv_scheme(deriv_name)
    hermition = gamma_hermition(gamma_name) * deriv_hermition(deriv_name)
    assert len(gamma) == 3 and len(deriv) == 3
    return [
        Operator([
            [1, gamma[1], deriv[2]],
            [1, gamma[2], deriv[1]],
        ], hermition),
        Operator([
            [1, gamma[2], deriv[0]],
            [1, gamma[0], deriv[2]],
        ], hermition),
        Operator([
            [1, gamma[0], deriv[1]],
            [1, gamma[1], deriv[0]],
        ], hermition),
    ]


def Q_ijk(gamma_name: GAMMA_NAME, deriv_name: DERIV_NAME):
    gamma = gamma_scheme(gamma_name)
    deriv = deriv_scheme(deriv_name)
    hermition = gamma_hermition(gamma_name) * deriv_hermition(deriv_name)
    assert len(gamma) == 3 and len(deriv) == 3
    return [
        Operator([
            [1, gamma[0], deriv[0]],
            [-1, gamma[1], deriv[1]],
        ], hermition),
        Operator([
            [-1, gamma[0], deriv[0]],
            [-1, gamma[1], deriv[1]],
            [2, gamma[2], deriv[2]],
        ], hermition),
    ]
