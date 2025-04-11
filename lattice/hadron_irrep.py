import sympy as sp
import numpy as np
from sympy import Matrix, I, S, Pow, Mul
from sympy.physics.quantum import Operator
from typing import Dict, List, Tuple, Literal

from opt_einsum import contract
from itertools import product

from .symmetry.utils import *
from .symmetry.gen_hardcoded_rep import (
    genLittleGroupIrrep,
    reductionToLittleGroup,
    wignerRotate,
)
from .symmetry.sympy_utils import *
from .quark_contract import *


class HadronIrrep(Operator):
    def __new__(cls, hadron_name: str, momentum: List[int], irrep_name: str, parity: int, tag: Tag):
        if parity is None:
            obj = super().__new__(cls, f"{hadron_name}({irrep_name}({tag.tag}),t={tag.time},{tuple(momentum)})")
        elif parity == -1:
            obj = super().__new__(cls, f"{hadron_name}({irrep_name}u({tag.tag}),t={tag.time},{tuple(momentum)},{parity})")
        else:
            obj = super().__new__(cls, f"{hadron_name}({irrep_name}g({tag.tag}),t={tag.time},{tuple(momentum)},{parity})")
        return obj

    def __init__(self, hadron_name: str, momentum: List[int], irrep_name: str, parity: int, tag: Tag):
        """
        Initialize a HadronIrrep object.

        Args:
            name: The name of the hadron
            momentum: The momentum of the hadron
            irrep_name: The irrep name
            parity: The parity
            tag: The tag
        """
        self.hadron_name = hadron_name
        self.momentum = momentum
        self.irrep_name = irrep_name
        self.parity = parity
        self.tag = tag

        if irrep_name.startswith("T"):
            self.lenth = 3
        elif irrep_name.startswith("G") or irrep_name.startswith("E"):
            self.lenth = 2
        elif irrep_name.startswith("H"):
            self.lenth = 4
        else:
            self.lenth = 1

    def __getitem__(self, row_idx):
        return HadronIrrepRow(self.hadron_name, self.momentum, self.irrep_name, row_idx, self.parity, self.tag)

    def __eq__(self, other):
        if not isinstance(other, HadronIrrep):
            return False
        else:
            return (
                self.hadron_name == other.hadron_name
                and self.momentum == other.momentum
                and self.irrep_name == other.irrep_name
                and self.parity == other.parity
                and self.tag == other.tag
            )

    def __hash__(self):
        return hash((self.hadron_name, tuple(self.momentum), self.irrep_name, self.parity, self.tag))


class HadronIrrepRow(Symbol):
    def __new__(cls, hadron_name: str, momentum: List[int], irrep_name: str, row_idx: int, parity: int, tag: Tag):
        if parity is None:
            obj = super().__new__(cls, f"{hadron_name}({irrep_name},{tuple(momentum)})[{row_idx}]", commutative=False)
        elif parity == -1:
            obj = super().__new__(cls, f"{hadron_name}({irrep_name}u,{tuple(momentum)})[{row_idx}]", commutative=False)
        else:
            obj = super().__new__(cls, f"{hadron_name}({irrep_name}g,{tuple(momentum)})[{row_idx}]", commutative=False)
        return obj

    def __init__(self, hadron_name: str, momentum: List[int], irrep_name: str, row_idx: int, parity: int, tag: Tag):
        """
        Initialize a HadronIrrepRow object.

        Args:
            name: The name of the hadron
            momentum: The momentum of the hadron
            irrep_name: The irrep name
            row_idx: The row index
            parity: The parity
            tag: The tag
        """
        self.hadron_name = hadron_name
        self.tag = tag
        self.momentum = momentum
        self.irrep_name = irrep_name
        self.row_idx = row_idx
        self.parity = parity
        self.rotate = genLittleGroupIrrep([0, 0, 0], "T_1", -1)
        self.little_group_matrix = genLittleGroupIrrep(momentum, irrep_name, parity, p_ref_irrep=True)

    def __eq__(self, other):
        if not isinstance(other, HadronIrrepRow):
            return False
        else:
            return (
                self.hadron_name == other.hadron_name
                and self.momentum == other.momentum
                and self.irrep_name == other.irrep_name
                and self.row_idx == other.row_idx
                and self.parity == other.parity
                and self.tag == other.tag
            )

    def __hash__(self):
        return hash((self.hadron_name, tuple(self.momentum), self.irrep_name, self.row_idx, self.parity, self.tag))

    def transform(self, group_element):
        momentum_final = list(self.rotate[group_element] @ Matrix(self.momentum))
        transform_matrix = self.little_group_matrix[wignerRotate(self.momentum, group_element)]
        result = S(0)
        for i in range(transform_matrix.shape[0]):
            result += transform_matrix[i, self.row_idx] * HadronIrrepRow(
                self.hadron_name, momentum_final, self.irrep_name, i, self.parity, self.tag
            )
        return result

    def __lt__(self, other):
        if self.tag.time != other.tag.time:
            return self.tag.time < other.tag.time
        if self.irrep_name != other.irrep_name:
            return self.irrep_name < other.irrep_name
        if self.row_idx != other.row_idx:
            return self.row_idx < other.row_idx
        return self.tag.tag < other.tag.tag

    def __gt__(self, other):
        return other < self

    def __le__(self, other):
        return self < other or self == other

    def __ge__(self, other):
        return self > other or self == other


from sympy import preorder_traversal


def transform_expression(expr, group_element):
    # 收集原表达式中的所有 HadronIrrepRow 实例
    instances = set()
    for sub_expr in preorder_traversal(expr):
        if isinstance(sub_expr, HadronIrrepRow):
            instances.add(sub_expr)

    # 创建替换映射：每个实例替换为其 transform 后的结果
    replacements = {inst: inst.transform(group_element) for inst in instances}

    # 应用替换并返回新表达式
    return expr.xreplace(replacements)


def expr_little_group_projection(expr, irrep_name, row_idx, parity=None):

    momentum = np.array([0, 0, 0], dtype=int)
    terms = Add.make_args(expr)
    factors = Mul.make_args(terms[0])
    for factor in factors:
        if isinstance(factor, HadronIrrepRow):
            momentum += np.array(factor.momentum, dtype=int)

    matrix_group = genLittleGroupIrrep(momentum, irrep_name, parity)
    len_irrep = matrix_group["iden"].shape[0]
    group_size = len(matrix_group.keys())
    projected_irrep_row = S(0)
    for key in matrix_group.keys():
        projected_irrep_row += (
            S(len_irrep) / S(group_size) * matrix_group[key][row_idx, row_idx] * transform_expression(expr, key)
        )
    return projected_irrep_row


def multi_exprs_little_group_projection(expr_list, irrep_name, row_idx, parity=None, single_result=False):
    result_expr_list = []
    for expr in expr_list:
        projected_expr = expr_little_group_projection(expr, irrep_name, row_idx, parity)
        if projected_expr != 0:
            if single_result:
                return projected_expr
            else:
                result_expr_list.append(projected_expr)
    # return result_expr_list
    return find_linear_independent_and_normalized_expr(result_expr_list)


def hadron_little_group_projection(hadrons, irrep_name, row_idx, parity=None, single_result=False):
    from itertools import product

    hadrons_list = []
    for i in range(len(hadrons)):
        hadron = hadrons[i]
        hadrons_list.append([hadron[j] for j in range(hadron.lenth)])

    exprs_mul_rows = []
    for expr in list(product(*hadrons_list)):
        exprs_mul_rows.append(Mul(*expr))
    return multi_exprs_little_group_projection(exprs_mul_rows, irrep_name, row_idx, parity, single_result)


if __name__ == "__main__":
    # edit the generator of fermion representation "generator.Fermion_generator"
    # generate the matrix representation of the group OHD with "genMatrixGroupOhD"

    # groupOhD = genMatrixGroupOhD(
    #     c4y=Fermion_generator["c4y"],
    #     c4z=Fermion_generator["c4z"],
    #     inv=Fermion_generator["inviden"],
    # )
    # print(groupOhD)

    # generate the multiplication table of the group OHD with "multiplicationTable"

    # groupOhD = Fermion_rep
    # print(groupOhD.keys())
    # print(multiplicationTable(groupOhD))

    # edit the generators of all little groups "OhD_generator","Dic4_generator","Dic3_generator","Dic2_generator","C4_generator1","C4_generator2" in generator.py
    # generate irreps of OHD with "genIrrepOhD"

    # OhD_irreps_Dict = {}
    # for key in ["A_1","A_2","E","T_1","T_2","G_1","G_2","H"]:
    #     print(key)
    #     OhD_irreps_Dict[key] = genIrrepOhD(key)
    # print(OhD_irreps_Dict)

    # generate irreps of all little groups with "genLittleGroupIrrep"

    # Dic4_irreps_dict={}
    # for key in ["A_1", "A_2", "E", "B1", "B2", "G_1", "G_2"]:
    #     Dic4_irreps_dict[key] = genLittleGroupIrrep([0, 0, 1], key,is_hardcoded=True)
    # print(Dic4_irreps_dict)

    # Dic2_irreps_dict = {}
    # for key in ["A_1", "A_2", "B1", "B2", "G"]:
    #     Dic2_irreps_dict[key] = genLittleGroupIrrep([0, 1, 1], key, is_hardcoded=False)
    # print(Dic2_irreps_dict)

    # Dic3_irreps_dict = {}
    # for key in ["A_1", "A_2", "F1", "F2",'E', "G"]:
    #     Dic3_irreps_dict[key] = genLittleGroupIrrep([1, 1, 1], key, is_hardcoded=False)
    # print(Dic3_irreps_dict)

    print(reductionToLittleGroup([0, 0, 1], "T_1", "A_1"))

    pass
