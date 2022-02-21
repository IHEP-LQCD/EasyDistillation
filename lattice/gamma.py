from .backend import getBackend

numpy = getBackend()

_one = numpy.identity(4)
_zero = numpy.zeros((4, 4))
_gamma_0 = numpy.array(
    [
        [0, 0, 0, 1j],
        [0, 0, 1j, 0],
        [0, -1j, 0, 0],
        [-1j, 0, 0, 0],
    ]
)
_gamma_1 = numpy.array(
    [
        [0, 0, 0, -1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
    ]
)
_gamma_2 = numpy.array(
    [
        [0, 0, 1j, 0],
        [0, 0, 0, -1j],
        [-1j, 0, 0, 0],
        [0, 1j, 0, 0],
    ]
)
_gamma_3 = numpy.array(
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
    return numpy.asarray((_gamma_0 if n & 0b0001 else _one) @ (_gamma_1 if n & 0b0010 else _one) @ (_gamma_2 if n & 0b0100 else _one) @ (_gamma_3 if n & 0b1000 else _one))
