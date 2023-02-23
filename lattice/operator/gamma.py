from functools import lru_cache

from ..backend import getBackend


class _Constant:
    @staticmethod
    @lru_cache(1)
    def zero():
        numpy = getBackend()
        return numpy.zeros((4, 4))

    @staticmethod
    @lru_cache(1)
    def one():
        numpy = getBackend()
        return numpy.identity(4)

    @staticmethod
    @lru_cache(1)
    def gamma_0():
        numpy = getBackend()
        return numpy.array([
            [0, 0, 0, 1j],
            [0, 0, 1j, 0],
            [0, -1j, 0, 0],
            [-1j, 0, 0, 0],
        ])

    @staticmethod
    @lru_cache(1)
    def gamma_1():
        numpy = getBackend()
        return numpy.array([
            [0, 0, 0, -1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
        ])

    @staticmethod
    @lru_cache(1)
    def gamma_2():
        numpy = getBackend()
        return numpy.array([
            [0, 0, 1j, 0],
            [0, 0, 0, -1j],
            [-1j, 0, 0, 0],
            [0, 1j, 0, 0],
        ])

    @staticmethod
    @lru_cache(1)
    def gamma_3():
        numpy = getBackend()
        return numpy.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])


def gamma(n: int):
    assert isinstance(n, int) and n >= 0 and n <= 15
    numpy = getBackend()
    return numpy.asarray(
        (_Constant.gamma_0() if n & 0b0001 else _Constant.one())
        @ (_Constant.gamma_1() if n & 0b0010 else _Constant.one())
        @ (_Constant.gamma_2() if n & 0b0100 else _Constant.one())
        @ (_Constant.gamma_3() if n & 0b1000 else _Constant.one())
    )


_naming_scheme = {
    "a0": [0],  # g0 0++
    "pi": [15],  # g5 0-+
    "pi(2)": [7],  # g5g4 0-+
    "b0": [8],  # g4 0+-
    "a1": [14, 13, 11],  # g5gi 1++
    "rho": [1, 2, 4],  # gi 1--
    "rho(2)": [9, 10, 12],  # gig4 1--
    "b1": [6, 5, 3],  # g5gig4 1+-
}

_naming_hermition = {
    "a0": "+",
    "pi": "-",
    "pi(2)": "+",
    "b0": "+",
    "a1": "-",
    "rho": "-",
    "rho(2)": "+",
    "b1": "-",
}

_naming_parity = {
    "a0": "+",
    "pi": "-",
    "pi(2)": "-",
    "b0": "+",
    "a1": "+",
    "rho": "-",
    "rho(2)": "-",
    "b1": "+",
}

_naming_charge_conjugation = {
    "a0": "+",
    "pi": "+",
    "pi(2)": "+",
    "b0": "-",
    "a1": "+",
    "rho": "-",
    "rho(2)": "-",
    "b1": "-",
}

_naming_time_reversal = {
    "a0": "+",
    "pi": "+",
    "pi(2)": "+",
    "b0": "+",
    "a1": "+",
    "rho": "+",
    "rho(2)": "+",
    "b1": "+",
}


def scheme(name: str):
    assert name in _naming_scheme
    return _naming_scheme[name]


def hermition(name: str):
    assert name in _naming_scheme
    return 1 if _naming_hermition[name] == "+" else -1


class GAMMA_NAME:
    A0 = "a0"
    B0 = "b0"
    PI = "pi"
    PI_2 = "pi(2)"
    RHO = "rho"
    RHO_2 = "rho(2)"
    A1 = "a1"
    B1 = "b1"


def instance(gamma_idxs: list):
    ret = _Constant.one()
    for idx in gamma_idxs:
        ret = ret @ gamma(idx)
    return ret
