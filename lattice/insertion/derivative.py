def output(derivative_coeff_index):
    c, n = derivative_coeff_index
    assert isinstance(n, int) and n >= 0
    d = ["dx", "dy", "dz"]
    ret = []
    num = -1
    while n >= 0:
        num += 1
        n = n - pow(3, num)
    n = n + pow(3, num)
    for _ in range(num):
        ret.append(d[n % 3])
        n = n // 3
    if c == 1:
        pass
    elif c == -1:
        ret.insert(0, "-")
    else:
        ret.insert(0, str(c))
    return "".join(ret)


def derivative(n: int):
    ret = []
    num = -1
    while n >= 0:
        num += 1
        n = n - pow(3, num)
    n = n + pow(3, num)
    for _ in range(num):
        ret.append(n % 3)
        n = n // 3
    return tuple(ret[::-1])


_naming_scheme = {
    "": [
        [[1, 0]],  # 1
    ],
    R"$\nabla$": [
        [[1, 1]],  # dx
        [[1, 2]],  # dy
        [[1, 3]],  # dz
    ],
    R"$\mathbb{B}$": [
        [[1, 11], [-1, 9]],  # dydz-dzdy
        [[1, 6], [-1, 10]],  # dzdx-dxdz
        [[1, 7], [-1, 5]],  # dxdy-dydx
    ],
    R"$\mathbb{D}$": [
        [[1, 11], [1, 9]],  # dydz+dzdy
        [[1, 6], [1, 10]],  # dzdx+dxdz
        [[1, 7], [1, 5]],  # dxdy+dydx
    ],
    R"$\mathbb{E}$": [
        [[1, 4], [-1, 8]],  # dxdx-dydy
        [[-1, 4], [-1, 8], [2, 12]],  # -dxdx-dydy+2dzdz
    ],
}

_naming_group = {
    "": "A_1",
    R"$\nabla$": "T_1",
    R"$\mathbb{B}$": "T_1",
    R"$\mathbb{D}$": "T_2",
    R"$\mathbb{E}$": "E",
}

_naming_hermiticity = {
    "": "+",
    R"$\nabla$": "-",
    R"$\mathbb{B}$": "-",
    R"$\mathbb{D}$": "+",
    R"$\mathbb{E}$": "+",
}

_naming_parity = {
    "": "+",
    R"$\nabla$": "-",
    R"$\mathbb{B}$": "+",
    R"$\mathbb{D}$": "+",
    R"$\mathbb{E}$": "+",
}

_naming_charge_conjugation = {
    "": "+",
    R"$\nabla$": "-",
    R"$\mathbb{B}$": "-",
    R"$\mathbb{D}$": "+",
    R"$\mathbb{E}$": "+",
}

_naming_time_reversal = {
    "": "+",
    R"$\nabla$": "+",
    R"$\mathbb{B}$": "+",
    R"$\mathbb{D}$": "+",
    R"$\mathbb{E}$": "+",
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


class DerivativeName:
    IDEN = ""
    NABLA = R"$\nabla$"
    B = R"$\mathbb{B}$"
    D = R"$\mathbb{D}$"
    E = R"$\mathbb{E}$"
