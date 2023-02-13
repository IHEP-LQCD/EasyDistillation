from typing import List, Union


class DerivPart:
    def __init__(self, coeff: float, deriv: str) -> None:
        self.coeff: float = coeff
        self.deriv: Union[int, str] = deriv

    def __repr__(self) -> str:
        return f"{self.coeff:.3f} * nabla_({self.deriv})"

    def normalize(self, sumsq):
        self.coeff *= sumsq**-0.5


class Deriv:
    def __init__(self, parts) -> None:
        sumsq = 0
        self.parts: List[DerivPart] = []
        for part in parts:
            sumsq += part[0]**2
            self.parts.append(DerivPart(part[0], part[1]))
        for part in self.parts:
            part.normalize(sumsq)

    def __repr__(self) -> str:
        ret = [part for part in self.parts]
        return f"{ret}"


_naming_scheme = {
    "_": [Deriv([[1, ""]]), ],
    "d": [
        Deriv([[1, "1"]]),
        Deriv([[1, "2"]]),
        Deriv([[1, "3"]]),
    ],
    "B": [
        Deriv([[1, "2 3"], [-1, "3 2"]]),
        Deriv([[1, "3 1"], [-1, "1 3"]]),
        Deriv([[1, "1 2"], [-1, "2 1"]]),
    ],
    "D": [
        Deriv([[1, "2 3"], [1, "3 2"]]),
        Deriv([[1, "3 1"], [1, "1 3"]]),
        Deriv([[1, "1 2"], [1, "2 1"]]),
    ],
    "E": [
        Deriv([[1, "1 1"], [-1, "2 2"]]),
        Deriv([[-1, "1 1"], [-1, "2 2"], [2, "3 3"]]),
    ],
}

_naming_hermition = {
    "_": "+",
    "d": "-",
    "B": "-",
    "D": "+",
    "E": "+",
}

_naming_parity = {
    "_": "+",
    "d": "-",
    "B": "+",
    "D": "+",
    "E": "+",
}

_naming_charge_conjugation = {
    "_": "+",
    "d": "-",
    "B": "-",
    "D": "+",
    "E": "+",
}

_naming_time_reversal = {
    "_": "+",
    "d": "+",
    "B": "+",
    "D": "+",
    "E": "+",
}


def scheme(name: str):
    assert name in _naming_scheme
    return _naming_scheme[name]


def hermition(name: str):
    assert name in _naming_scheme
    return 1 if _naming_hermition[name] == "+" else -1


class DERIV_NAME:
    pass
