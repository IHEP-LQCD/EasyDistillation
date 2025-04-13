import numpy as np
from itertools import permutations
import sympy as sp
from sympy import Matrix, sqrt, Pow
from sympy import Add, Mul, Expr, Symbol
from sympy.physics.quantum import Operator


def split_first_term(expr: Expr):
    """
    Split expression by sum, return first element.
    """
    if isinstance(expr, Add):
        # If addition expression, return first term
        return expr.args[0]
    else:
        # If not addition expression, return expression itself
        return expr


def split_mul(expr: Expr):
    """
    Split expression by product, return list of all terms.
    """
    if isinstance(expr, Mul):
        # If multiplication expression, return all terms
        return expr.args
    elif isinstance(expr, Pow) and isinstance(expr, sp.core.power.Pow) and expr.args[1] == 2:
        return [expr.args[0], expr.args[0]]
    else:
        # If not multiplication expression, return list containing expression itself
        return [expr]


def split_expression(expr: Expr):
    """
    Split expression by sum to get first element, then split by product.
    """
    # Split by sum to get first element
    first_term = split_first_term(expr)
    # Split first element by product
    mul_terms = split_mul(first_term)
    return mul_terms


from collections import defaultdict
from sympy import Expr, Mul, Add, S
from sympy.matrices import Matrix


def collect_product_basis(exprs: list[Expr]) -> list[Expr]:
    """Collect all unique product terms as basis vectors"""
    basis_map = {}  # {frozenset(product_factors): index}

    for expr in exprs:
        terms = Add.make_args(expr)
        for term in terms:
            factors = Mul.make_args(term)
            coeff = S(1)
            product_factors = []
            for factor in factors:
                if isinstance(factor, Symbol):
                    product_factors.append(factor.name)
                else:
                    coeff *= factor
            # coeff, factors = term.as_coeff_mul()
            # Extract Operator product combinations
            product_factors = tuple(product_factors)
            # Assign basis vector index to each unique product
            key = product_factors
            if key not in basis_map:
                basis_map[key] = None
    return list(basis_map.keys())


def build_coefficient_matrix(exprs: list[Expr], basis_map: dict) -> list[list]:
    """Build coefficient matrix (considering product terms)"""
    matrix = []

    for expr in exprs:
        row = [S(0)] * len(basis_map)  # Use SymPy's S(0) instead of 0.0
        terms = Add.make_args(expr)

        for term in terms:
            factors = Mul.make_args(term)
            coeff = S(1)
            product_factors = []

            for factor in factors:
                if isinstance(factor, Symbol):
                    product_factors.append(factor.name)
                else:
                    coeff *= factor

            product_factors = tuple(product_factors)

            if product_factors in basis_map:
                idx = basis_map[product_factors]
                row[idx] += coeff  # Use SymPy coefficients for exact computation

        matrix.append(row)

    return matrix


def find_linear_independent_exprs(exprs: list[Expr]) -> list[Expr]:
    # Filter zero expressions - use safer way to check if zero
    # Use direct comparison instead of is_zero property
    filtered = []
    for expr in exprs:
        try:
            if expr != 0:  # Use !=0 instead of not expr.is_zero
                filtered.append(expr)
        except Exception:
            # If comparison fails, assume expression is non-zero
            filtered.append(expr)

    if not filtered:
        return []

    # Collect basis vectors
    basis = collect_product_basis(filtered)
    basis_map = {vec: idx for idx, vec in enumerate(basis)}

    # Build coefficient matrix
    coeff_matrix = build_coefficient_matrix(filtered, basis_map)
    # Convert coefficient matrix to SymPy matrix for Gaussian elimination
    M = Matrix(coeff_matrix)

    # Execute Row Reduced Echelon Form (RREF)
    rref_matrix, pivots = M.rref()

    # Find indices of linearly independent expressions
    independent_indices = []
    for i, has_pivot in enumerate(pivots):
        if i < len(filtered):
            independent_indices.append(i)

    # Extract linearly independent expressions from the original list
    independent_exprs = [filtered[i] for i in independent_indices]

    # If no linearly independent expressions found, return empty list
    if not independent_exprs:
        return []

    return independent_exprs


def convert_pow_to_mul(expr):
    """
    Convert all power operations in sympy expression to product form

    Parameters:
        expr: sympy expression

    Returns:
        Converted expression
    """
    if expr.is_Pow:
        base, exp = expr.args
        if exp.is_Integer and exp > 0:
            # Convert power operation to continuous multiplication
            return Mul(*[convert_pow_to_mul(base) for _ in range(exp)], evaluate=False)
        return expr
    elif expr.is_Atom or isinstance(expr, Operator):
        return expr
    else:
        # Recursively process all arguments
        return expr.func(*[convert_pow_to_mul(arg) for arg in expr.args])
