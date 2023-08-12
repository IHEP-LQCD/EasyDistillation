# Chunjiang Shi
# May. 2022
# Construct two-particle group representation
# citation:
# https://doi.org/10.1007/JHEP01(2017)129
# https://arxiv.org/abs/1607.06738 eq.(4.11)
import numpy as np
import sympy as sp
from sympy import S
from sympy.physics.quantum import Operator
from sympy.physics.quantum.cg import CG
from sympy.functions.special.spherical_harmonics import Ynm
from itertools import product


def list_from_mom2_max(n):
    imax = int(np.sqrt(n))
    mom = []
    for it in product(range(-imax, imax + 1), repeat=3):
        i, j, k = it
        if i**2 + j**2 + k**2 <= n:
            if i**2 + j**2 + k**2 == n:
                mom.append([S(k), S(j), S(i)])
                # print(f"mom list:({k},{j},{i})")
    return mom


def make_operator(form, mom="p"):
    if form[0] == "P":
        return [Operator(form + f"({mom})")]
    elif form[0] == "V":
        return [
            sp.I * Operator(f"{form}_z({mom})"),  # m = 0
            -sp.I
            * (Operator(f"{form}_x({mom})") + sp.I * Operator(f"{form}_y({mom})"))
            / sp.sqrt(2),  # m = 1
            sp.I
            * (Operator(f"{form}_x({mom})") - sp.I * Operator(f"{form}_y({mom})"))
            / sp.sqrt(2),  # m = -1
        ]


def rotation(vec):
    x, y, z = vec
    r = sp.sqrt(x**2 + y**2 + z**2)
    if r == 0:
        return 0, 0
    theta = sp.acos(z / r)
    phi = sp.atan2(y, x)
    if phi == sp.nan:
        phi = 0
    # print(vec, "new : ", theta, phi)
    return theta, phi


def two_particle_circle_basis_JM(op1, op2, mom2, J, M, L, Spin):
    S1 = 0 if op1[0] == "P" else 1
    S2 = 0 if op2[0] == "P" else 1
    momentum_list = list_from_mom2_max(mom2)
    basis = 0
    for s1 in range(-S1, S1 + 1):
        for s2 in range(-S2, S2 + 1):
            mS = s1 + s2
            for mL in range(-L, L + 1):
                for ip in momentum_list:
                    theta, phi = rotation(ip)
                    basis += (
                        CG(S(L), S(mL), S(Spin), S(mS), J, M)
                        * CG(S1, s1, S2, s2, S(Spin), mS)
                        * sp.simplify(Ynm(S(L), S(mL), theta, phi)).expand(func=True)
                        * make_operator(op1, f"{ip}")[s1]
                        * make_operator(op2, f"{[-ii for ii in ip]}")[s2]
                    )
    print(sp.expand(sp.simplify(basis)))
    return sp.simplify(basis)


def two_particle_circle_basis(op1, op2, mom2, J, L, Spin):
    """
    return opetator[-J, -J+1, ..., +J]
    """
    S1 = 0 if op1[0] == "P" else 1
    S2 = 0 if op2[0] == "P" else 1
    momentum_list = list_from_mom2_max(mom2)
    basis = [0] * (2 * J + 1)
    for M in range(-J, J + 1):
        ans = 0
        for s1 in range(-S1, S1 + 1):
            for s2 in range(-S2, S2 + 1):
                mS = s1 + s2
                for mL in range(-L, L + 1):
                    for ip in momentum_list:
                        theta, phi = rotation(ip)
                        # print(
                        #     CG(S(L), S(mL), S(Spin), S(mS), J, M),
                        #     CG(S(L), S(mL), S(Spin), S(mS), J, M).doit(), CG(S_op1, is1, S_op2, is2, S(Spin), mS),
                        #     CG(S_op1, is1, S_op2, is2, S(Spin), mS).doit()
                        # )
                        # print(sp.simplify(Ynm(S(L), S(mL), theta, phi)).expand(func=True))
                        ans += (
                            CG(S(L), S(mL), S(Spin), S(mS), J, M)
                            * CG(S1, s1, S2, s2, S(Spin), mS)
                            * sp.simplify(Ynm(S(L), S(mL), theta, phi)).expand(
                                func=True
                            )
                            * make_operator(op1, f"{ip}")[s1]
                            * make_operator(op2, f"{[-ii for ii in ip]}")[s2]
                        )
        # print(sp.expand(sp.simplify(ans)))
        basis[M + J] = sp.expand(sp.simplify(ans))
    return basis


def two_particle_Cartesian_basis(op1, op2, mom2, J, L, Spin):
    """
    return opetator[row1, row2, row3] in O irrep.
    """
    circle_basis = two_particle_circle_basis(op1, op2, mom2, J, L, Spin)
    if J == 0:
        return circle_basis
    elif J == 1:
        return [
            sp.expand(
                sp.simplify((circle_basis[0] - circle_basis[2]) * (-sp.I) / sp.sqrt(2))
            ),
            sp.expand(sp.simplify((circle_basis[0] + circle_basis[2]) / sp.sqrt(2))),
            sp.expand(sp.simplify(circle_basis[1] * (-sp.I))),
        ]
    else:
        raise NotImplementedError("TODO: J>1")
