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
from .hadron_irrep import HadronIrrepRow, HadronIrrep
from .quark_diagram import diagram_simplify


class Hadron:
    def __init__(self, irep_row: Expr, flavor_structure: Expr):
        self.irrep_row = irep_row
        self.flavor_structure = flavor_structure

    def set_time(self, time: int) -> "Hadron":
        """
        Modify the time value of all Tags in the Hadron.
        Traverse the expression to find HadronIrrepRow and MesonFlavorStructure leaf nodes,
        and replace their tag.time value with the specified time value

        Args:
            time: The new time value to set

        Returns:
            A new Hadron instance
        """
        # Modify the tag in irrep_row
        new_irrep_row = set_time_in_expr(self.irrep_row, time)

        # Modify the tag in flavor_structure
        new_flavor_structure = set_time_in_expr(self.flavor_structure, time)

        # Return a new Hadron instance
        return Hadron(new_irrep_row, new_flavor_structure)


def set_time_in_expr(expr: Expr, time: int) -> Expr:
    """
    Recursively traverse the expression to modify the tag.time value of leaf nodes

    Args:
        expr: The expression to modify
        time: The new time value to set

    Returns:
        The modified expression
    """
    replacements = {}

    # Traverse all sub-expressions in the expression
    for sub_expr in sp.preorder_traversal(expr):
        if isinstance(sub_expr, HadronIrrepRow):
            # Create a new Tag, only modify the time value
            new_tag = Tag(sub_expr.tag.tag, time)
            # Create a new HadronIrrepRow instance
            new_row = HadronIrrepRow(
                sub_expr.name, sub_expr.momentum, sub_expr.irrep_name, sub_expr.row_idx, sub_expr.parity, new_tag
            )
            replacements[sub_expr] = new_row
        elif isinstance(sub_expr, HadronFlavorStructure):
            # For MesonFlavorStructure, create a new instance
            new_meson = HadronFlavorStructure(sub_expr.flavor_str, time)
            replacements[sub_expr] = new_meson

    # Apply replacements and return the new expression
    return expr.xreplace(replacements)


def set_time_in_list(hadron_list: List[Expr], time: int) -> Expr:
    """
    Apply set_time_in_expr function to each expression in the list, and set_time function to each Hadron
    """
    hadron_list_new = []
    for i in range(len(hadron_list)):
        hadron = hadron_list[i]
        if isinstance(hadron, Hadron):
            hadron_list_new.append(hadron.set_time(time))
        else:
            hadron_list_new.append(set_time_in_expr(hadron, time))
    return hadron_list_new


def gen_correlator(hadrons: List[List[Hadron]], time_slice_list=None):
    """
    Calculate correlation functions for multiple Hadron lists

    Args:
        hadrons: List containing multiple lists of Hadron objects
        time_slice_list: List of time slices
        operator_list: List of operators

    Returns:
        Correlation function
    """
    if time_slice_list is None:
        time_slice_list = [i for i in range(len(hadrons))]
    for i in range(len(hadrons)):
        hadrons[i] = set_time_in_list(hadrons[i], time_slice_list[i])
    result_matrix = np.ndarray(shape=tuple([len(hadrons[i]) for i in range(len(hadrons))]), dtype=object)
    # Create all possible index combinations
    indices_ranges = [range(len(hadrons[i])) for i in range(len(hadrons))]
    indices_combinations = list(product(*indices_ranges))
    # Iterate through all index combinations
    for indices in indices_combinations:
        # Build corresponding hadron_tuple
        hadron_tuple = tuple(hadrons[i][indices[i]] for i in range(len(hadrons)))
        position_wavefnc = convert_pow_to_mul(
            Mul(*[hadron_tuple[i].irrep_row for i in range(len(hadron_tuple))]).expand()
        )
        flavor_wavefnc = convert_pow_to_mul(
            Mul(*[hadron_tuple[i].flavor_structure for i in range(len(hadron_tuple))]).expand()
        )
        result = S(0)
        terms = Add.make_args(position_wavefnc)
        for term in terms:
            insersion_list = []
            factors = Mul.make_args(term)
            num_hadrons = 0
            for factor in factors:
                if isinstance(factor, HadronIrrepRow):
                    insersion_list.append(factor)
                    num_hadrons += 1
            result += diagram_simplify(quark_contract(flavor_wavefnc, insersion_list, degenerate=True))

        # Store hadron_tuple in corresponding position in result
        result_matrix[indices] = sp.simplify(result)
    return result_matrix
