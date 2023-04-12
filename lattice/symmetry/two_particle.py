# construct two-particle group represention
# from arXiv: 1607.06738  eq.(4.11)
# @Chunjiang. Shi.
# May. 2022.
import numpy as np
import sympy as sp
from sympy import S
from sympy.physics.quantum import Operator
from sympy.physics.quantum.cg import CG
from sympy.functions.special.spherical_harmonics import Ynm
from itertools import product


def momlist(n):
    imax = int(np.sqrt(n))
    mom = []
    for it in product(range(-imax, imax + 1), repeat=3):
        i, j, l = it
        if (i**2 + j**2 + l**2 <= n):
            if (i**2 + j**2 + l**2 == n):
                mom.append([S(l), S(j), S(i)])
                # print(f"mom list:({l},{j},{i})")
    return mom


def makeOperater(form, mom="p"):
    if form[0] == "P":
        return [Operator(form + f"({mom})")]
    elif form[0] == "V":
        return [
            sp.I * Operator(f"{form}_z({mom})"),  # m = 0
            -sp.I *(Operator(f"{form}_x({mom})") + sp.I * Operator(f"{form}_y({mom})")) / sp.sqrt(2),  # m = 1
            sp.I * (Operator(f"{form}_x({mom})") - sp.I * Operator(f"{form}_y({mom})")) /sp.sqrt(2),   # m = -1
        ]# yapf:disable

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


def two_particle_operator_circle_basis_JM(op1, op2, p2, J, M, L, Spin):
    S_op1 = 0 if op1[0] == "P" else 1
    S_op2 = 0 if op2[0] == "P" else 1
    pList = momlist(p2)
    ans = 0
    for is1 in range(-S_op1, S_op1 + 1):
        for is2 in range(-S_op2, S_op2 + 1):
            mS = is1 + is2
            for mL in range(-L, L + 1):
                for ip in pList:
                    theta, phi = rotation(ip)
                    ans += CG(S(L), S(mL), S(Spin), S(mS),
                              J, M) * CG(S_op1, is1, S_op2, is2, S(Spin), mS) * sp.simplify(
                                  Ynm(S(L), S(mL), theta, phi)
                              ).expand(func=True
                                      ) * makeOperater(op1, f"{ip}")[is1] * makeOperater(op2,
                                                                                         f"{[-ii for ii in ip]}")[is2]
    print(sp.expand(sp.simplify(ans)))
    return sp.simplify(ans)


def two_particle_operator_circle_basis(op1, op2, p2, J, L, Spin):
    '''
    return opetator[-J, -J+1, ..., +J]
    '''
    S_op1 = 0 if op1[0] == "P" else 1
    S_op2 = 0 if op2[0] == "P" else 1
    pList = momlist(p2)
    ret = [0] * (2 * J + 1)
    for M in range(-J, J + 1):
        ans = 0
        for is1 in range(-S_op1, S_op1 + 1):
            for is2 in range(-S_op2, S_op2 + 1):
                mS = is1 + is2
                for mL in range(-L, L + 1):
                    for ip in pList:
                        theta, phi = rotation(ip)
                        # print( CG(S(L),S(mL),S(Spin), S(mS), J, M),  CG(S(L),S(mL),S(Spin), S(mS), J, M).doit(), CG(S_op1, is1, S_op2, is2, S(Spin), mS) , CG(S_op1, is1, S_op2, is2, S(Spin), mS).doit())
                        # print(sp.simplify(Ynm(S(L), S(mL), theta, phi)).expand(func=True))
                        ans += CG(S(L), S(mL), S(Spin), S(mS), J,
                                  M) * CG(S_op1, is1, S_op2, is2, S(Spin), mS) * sp.simplify(
                                      Ynm(S(L), S(mL), theta, phi)
                                  ).expand(
                                      func=True
                                  ) * makeOperater(op1, f"{ip}")[is1] * makeOperater(op2, f"{[-ii for ii in ip]}")[is2]
        # print(sp.expand(sp.simplify(ans)))
        ret[M + J] = sp.expand(sp.simplify(ans))
    return ret


def two_particle_operator_Cartesian_basis(op1, op2, p2, J, L, Spin):
    '''
    return opetator[row1, row2, row3] in O irrep.
    '''
    ret = two_particle_operator_circle_basis(op1, op2, p2, J, L, Spin)
    if J == 0:
        return ret
    elif J == 1:
        return [
            sp.expand(sp.simplify((ret[0] - ret[2]) * (-sp.I)/ sp.sqrt(2))),
            sp.expand(sp.simplify((ret[0] + ret[2])  / sp.sqrt(2))),
            sp.expand(sp.simplify(ret[1] * (-sp.I))),
        ]
    else:
        raise NotImplementedError("TODO: J>1")
