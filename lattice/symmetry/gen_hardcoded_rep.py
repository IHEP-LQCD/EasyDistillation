import sympy as sp
import numpy as np
from sympy import Matrix, I, S
from sympy.physics.quantum import Operator
from .hardcoded_rep import *
from typing import Dict, List, Tuple, Literal
from .utils import *
from .group_generator import *
from opt_einsum import contract
import copy
from itertools import product


def genMatrixGroupOhD(c4y: Matrix, c4z: Matrix, inv: Matrix = None):
    """
    Generate the irrep of group O with given generators c4y and c4z
    """
    iden = c4y.inv() @ c4y
    c4x = c4z.inv() @ c4y @ c4z
    c3 = c4x @ c4y
    c3x = (c4z @ c4y).inv()
    c3y = (c4x @ c4z).inv()
    c3z = (c4y @ c4x).inv()
    group = {
        "iden": iden,
        "c4x": c4x,
        "c2x": c4x @ c4x,
        "c4x^-1": c4x.inv(),
        "c4y": c4y,
        "c2y": c4y @ c4y,
        "c4y^-1": c4y.inv(),
        "c4z": c4z,
        "c2z": c4z @ c4z,
        "c4z^-1": c4z.inv(),
        "c3delta": c3,
        "c3delta^-1": c3.inv(),
        "c3gamma": c3x,
        "c3gamma^-1": c3x.inv(),
        "c3beta": c3y,
        "c3beta^-1": c3y.inv(),
        "c3alpha": c3z,
        "c3alpha^1": c3z.inv(),
        "c2e": c4y @ c4z @ c4y,
        "c2f": c4y @ c4y @ c4x,
        "c2c": c4y @ c4z @ c4z,
        "c2d": c4z @ c4z @ c4y,
        "c2a": c4y @ c4y @ c4z,
        "c2b": c4x @ c4x @ c4z,
    }
    for key in group.keys():
        group[key] = group[key].applyfunc(sp.simplify)
    group_tmp = group.copy()
    r_2pi = sp.simplify(c4x @ c4x @ c4x @ c4x)
    for key in group_tmp.keys():
        group[f"r{key}"] = r_2pi @ group_tmp[key]

    group_tmp = group.copy()
    if inv is not None:
        for key in group_tmp.keys():
            group[f"inv{key}"] = inv @ group_tmp[key]
    for key in group.keys():
        group[key] = group[key].applyfunc(sp.S)
        group[key] = group[key].applyfunc(sp.simplify)
    return group


def genIrrepOhD(
    irrep_name: Literal["A_1", "A_2", "E", "T_1", "T_2", "G_1", "G_2", "H"],
    parity: Literal[1, -1, None] = None,
):
    """
    Generate the group element of the irrep.
    """
    generator_irrep = OhD_generator[irrep_name]
    c4y = generator_irrep["c4y"]
    c4z = generator_irrep["c4z"]
    if parity is not None:
        parity = Matrix(parity * np.eye(c4y.shape[0]))
    return genMatrixGroupOhD(c4y, c4z, parity)


def littleGroup(fixed_point=[0, 0, 0], group=None, elem=True):
    """
    Generate the little group element which keep the "fixed_point" fixed.
    """
    if group is None:
        group = genIrrepOhD("T_1", -1)
    little_group_element = {}
    fixed_point = Matrix(fixed_point)
    for key in group.keys():
        if (group[key] @ fixed_point - fixed_point).norm() < 0.01:
            if elem:
                little_group_element[key] = group[key]
            else:
                little_group_element[key] = None
    return little_group_element


def momentunSymplify(p):
    """
    p of the same little group is simplified to a representative element
    """
    p = [int(i) for i in p]
    classification = {0: []}
    for i in range(3):
        if abs(p[i]) in classification:
            classification[abs(p[i])].append(i)
        else:
            classification[abs(p[i])] = [i]
    p = [np.sign(ele) for ele in p]
    doublekey = 0
    singlekey = []
    for key in classification.keys():
        if len(classification[key]) != 1 and key != 0:
            doublekey = 1
        else:
            singlekey.append(key)
    for i in range(len(singlekey)):
        for index in classification[np.sort(singlekey)[i]]:
            p[index] *= i + doublekey
    return p


def genR_ref(p_ref, group=None, all=False):
    """
    Generate the R_ref that relate and reference momentum p_ref and any momentum. Stashed after hardcoding.
    """
    if group == None:
        group = genIrrepOhD("T_1", -1)
    wigner_dict = {}
    for ele in group.keys():
        pf = group[ele] @ Matrix(p_ref)
        pf_str = ",".join([str(element) for element in pf])
        if all:
            if pf_str not in wigner_dict.keys():
                wigner_dict[pf_str] = [ele]
            else:
                wigner_dict[pf_str].append(ele)
        else:
            if pf_str not in wigner_dict.keys():
                wigner_dict[pf_str] = ele
    return wigner_dict


def wignerRotate(p_i: list, ele: str):
    """
    Rotate wigner rotation for ele with initial momentum pi.
    """
    rotation = genIrrepOhD("T_1", -1)
    p_i = momentunSymplify(p_i)
    p_i_str = ",".join([str(element) for element in p_i])
    p_f = sp.sympify(rotation[ele] @ Matrix(p_i))
    p_f_str = ",".join([str(int(element)) for element in p_f])
    if p_i_str in refRotateDict.keys():
        ref_rotate_p_f_inv = OhD_inv(refRotateDict[p_f_str])
        result = OhD_mul(OhD_mul(ref_rotate_p_f_inv, ele), refRotateDict[p_i_str])
        return result
    else:
        raise NotImplementedError(f"{p_i_str} not finished")


def genLittleGroupIrrep(p, irrep_name, parity=None, p_ref=None, is_hardcoded=True, p_ref_irrep=False):
    """
    Generate the irrep of the little group with given generator and momentum p. Will be stashed after hardcoding the irrep of all little groups.
    """
    p = momentunSymplify(p)
    if sum([ele**2 for ele in p]) == 0:
        p_ref = [0, 0, 0]
        generator = OhD_generator[irrep_name]
        hardcode_irrep = OD_irreps[irrep_name].copy()
        if parity is not None:
            generator["inviden"] = Matrix(parity * np.eye(generator["c4y"].shape[0]))
            hardcode_irrep_tmp = copy.deepcopy(hardcode_irrep)
            for key in hardcode_irrep_tmp.keys():
                hardcode_irrep[f"inv{key}"] = parity * hardcode_irrep_tmp[key]
    elif sum([ele**2 for ele in p]) == 1:
        p_ref = [0, 0, 1]
        generator = Dic4_generator[irrep_name]
        hardcode_irrep = Dic4_irreps[irrep_name]
    elif sum([ele**2 for ele in p]) == 2:
        p_ref = [0, 1, 1]
        generator = Dic2_generator[irrep_name]
        hardcode_irrep = Dic2_irreps[irrep_name]
    elif sum([ele**2 for ele in p]) == 3:
        p_ref = [1, 1, 1]
        generator = Dic3_generator[irrep_name]
        hardcode_irrep = Dic3_irreps[irrep_name]
    elif sum([ele**2 for ele in p]) == 5:
        p_ref = [0, 1, 2]
        generator = C4_generator2[irrep_name]
        is_hardcoded = False
    elif sum([ele**2 for ele in p]) == 6:
        p_ref = [2, 1, 1]
        generator = C4_generator1[irrep_name]
        is_hardcoded = False
    else:
        raise NotImplementedError(f"p^2={p} not implemented")
    if is_hardcoded:
        little_group_pref = hardcode_irrep
    else:
        little_group_pref = littleGroup(p_ref, elem=False)
        nkey = len(little_group_pref.keys()) - len(generator.keys())
        for key in generator.keys():
            generator[key] = Matrix(generator[key])
            little_group_pref[key] = generator[key]
        while nkey > 0:
            for key1 in little_group_pref.keys():
                if little_group_pref[key1] is not None:
                    for key2 in generator.keys():
                        key_result = OhD_mul(key1, key2)
                        if little_group_pref[key_result] is None:
                            little_group_pref[key_result] = (little_group_pref[key1] @ generator[key2]).applyfunc(
                                sp.simplify
                            )
                            nkey -= 1
    if p_ref == [0, 0, 0] or p_ref_irrep:
        little_group_p = little_group_pref
    else:
        little_group_p = littleGroup(p, elem=False)
        for key in little_group_p.keys():
            wignerRotated_key = wignerRotate(p, key)
            little_group_p[key] = little_group_pref[wignerRotated_key]
    return little_group_p


def reductionToLittleGroup(momentum, OhD_irep_name, parity, little_group_irrep_name):
    """
    Reduce the OhD irrep to the little group irrep.
    """
    p = momentunSymplify(momentum)
    matrix_little_group_irrep = genLittleGroupIrrep(p, little_group_irrep_name)
    matrix_OhD_irrep = genIrrepOhD(OhD_irep_name, parity)
    ndim_little_group = matrix_little_group_irrep["iden"].shape[0]
    ndim_OhD = matrix_OhD_irrep["iden"].shape[0]
    reduction_matrix_colinear = np.full((ndim_little_group, ndim_OhD, ndim_OhD), S(0))
    reduction_matrix = np.full((ndim_little_group, ndim_OhD), S(0))
    for key in matrix_little_group_irrep.keys():
        reduction_matrix_colinear += np.einsum(
            "ij,kk->kji",
            np.array(matrix_OhD_irrep[key]),
            np.array(matrix_little_group_irrep[key]),
        )
    for i in range(ndim_little_group):
        reduction_matrix[i] = check_and_normalize_arrays(reduction_matrix_colinear[i])
    return reduction_matrix
