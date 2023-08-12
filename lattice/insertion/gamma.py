from functools import lru_cache

from ..backend import get_backend


class _Constant:
    @staticmethod
    @lru_cache(1)
    def zero():
        backend = get_backend()
        return backend.zeros((4, 4))

    @staticmethod
    @lru_cache(1)
    def one():
        backend = get_backend()
        return backend.identity(4)

    @staticmethod
    @lru_cache(1)
    def gamma_0():
        backend = get_backend()
        return backend.array(
            [
                [0, 0, 0, 1j],
                [0, 0, 1j, 0],
                [0, -1j, 0, 0],
                [-1j, 0, 0, 0],
            ]
        )

    @staticmethod
    @lru_cache(1)
    def gamma_1():
        backend = get_backend()
        return backend.array(
            [
                [0, 0, 0, -1],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
            ]
        )

    @staticmethod
    @lru_cache(1)
    def gamma_2():
        backend = get_backend()
        return backend.array(
            [
                [0, 0, 1j, 0],
                [0, 0, 0, -1j],
                [-1j, 0, 0, 0],
                [0, 1j, 0, 0],
            ]
        )

    @staticmethod
    @lru_cache(1)
    def gamma_3():
        backend = get_backend()
        return backend.array(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ]
        )


def output(n: int):
    assert isinstance(n, int) and 0 <= n <= 15
    if n == 0:
        return ""
    elif n == 15:
        return "γ5"
    elif n == 7:
        return "γ4γ5"
    elif n == 8:
        return "γ4"
    elif n in [14, 13, 11]:
        return f"γ5γ{[14, 13, 11].index(n)+1}"
    elif n in [1, 2, 4]:
        return f"γ{[1, 2, 4].index(n)+1}"
    elif n in [9, 10, 12]:
        return f"γ4γ{[9, 10, 12].index(n)+1}"
    elif n in [6, 5, 3]:
        return f"γ4γ5γ{[6, 5, 3].index(n)+1}"


def gamma(n: int):
    assert isinstance(n, int) and 0 <= n <= 15
    backend = get_backend()
    return backend.asarray(
        (_Constant.gamma_0() if n & 0b0001 else _Constant.one())
        @ (_Constant.gamma_1() if n & 0b0010 else _Constant.one())
        @ (_Constant.gamma_2() if n & 0b0100 else _Constant.one())
        @ (_Constant.gamma_3() if n & 0b1000 else _Constant.one())
    )


_naming_scheme = {
    R"$a_0$": [0],  # g0 0++
    R"$\pi$": [15],  # g5 0-+
    R"$\pi(2)$": [7],  # g5g4 0-+
    R"$b_0$": [8],  # g4 0+-
    R"$a_1$": [14, 13, 11],  # g5gi 1++
    R"$\rho$": [1, 2, 4],  # gi 1--
    R"$\rho(2)$": [9, 10, 12],  # gig4 1--
    R"$b_1$": [6, 5, 3],  # g5gig4 1+-
}

_naming_group = {
    R"$a_0$": "A_1",
    R"$\pi$": "A_1",
    R"$\pi(2)$": "A_1",
    R"$b_0$": "A_1",
    R"$a_1$": "T_1",
    R"$\rho$": "T_1",
    R"$\rho(2)$": "T_1",
    R"$b_1$": "T_1",
}

_naming_hermiticity = {
    R"$a_0$": "+",
    R"$\pi$": "-",
    R"$\pi(2)$": "+",
    R"$b_0$": "+",
    R"$a_1$": "-",
    R"$\rho$": "-",
    R"$\rho(2)$": "+",
    R"$b_1$": "-",
}

_naming_parity = {
    R"$a_0$": "+",
    R"$\pi$": "-",
    R"$\pi(2)$": "-",
    R"$b_0$": "+",
    R"$a_1$": "+",
    R"$\rho$": "-",
    R"$\rho(2)$": "-",
    R"$b_1$": "+",
}

_naming_charge_conjugation = {
    R"$a_0$": "+",
    R"$\pi$": "+",
    R"$\pi(2)$": "+",
    R"$b_0$": "-",
    R"$a_1$": "+",
    R"$\rho$": "-",
    R"$\rho(2)$": "-",
    R"$b_1$": "-",
}

_naming_time_reversal = {
    R"$a_0$": "+",
    R"$\pi$": "+",
    R"$\pi(2)$": "+",
    R"$b_0$": "+",
    R"$a_1$": "+",
    R"$\rho$": "+",
    R"$\rho(2)$": "+",
    R"$b_1$": "+",
}


def scheme(name: str):
    assert name in _naming_scheme
    return _naming_scheme[name]


def group(name: str):
    assert name in _naming_scheme
    return _naming_group[name]


def parity(name: str):
    assert name in _naming_scheme
    return 1 if _naming_parity[name] == "+" else -1


def charge_conjugation(name: str):
    assert name in _naming_scheme
    return 1 if _naming_charge_conjugation[name] == "+" else -1


def hermiticity(name: str):
    assert name in _naming_scheme
    return 1 if _naming_hermiticity[name] == "+" else -1


class GammaName:
    A0 = R"$a_0$"
    B0 = R"$b_0$"
    PI = R"$\pi$"
    PI_2 = R"$\pi(2)$"
    RHO = R"$\rho$"
    RHO_2 = R"$\rho(2)$"
    A1 = R"$a_1$"
    B1 = R"$b_1$"
