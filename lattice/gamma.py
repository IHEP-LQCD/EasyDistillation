from functools import lru_cache


from .backend import getBackend


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
        return numpy.array(
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
        numpy = getBackend()
        return numpy.array(
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
        numpy = getBackend()
        return numpy.array(
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
        numpy = getBackend()
        return numpy.array(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ]
        )


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
    "a0": [[0]],
    "a0(2)": [[8]],
    "pi": [[15]],
    "pi(2)": [[8, 15]],
    "rho": [[1], [2], [4]],
    "rho(2)": [[8, 1], [8, 2], [8, 4]],
    "a1": [[15, 1], [15, 2], [15, 4]],
    "b1": [[8, 15, 1], [8, 15, 2], [8, 15, 4]],
}


def scheme(name: str):
    assert name in _naming_scheme
    return _naming_scheme[name]


class GAMMA_NAME:
    A0 = "a0"
    A0_2 = "a0(2)"
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
