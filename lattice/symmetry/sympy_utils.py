import numpy as np
from itertools import permutations
import sympy as sp
from sympy import Matrix, sqrt, Pow
from sympy import Add, Mul, Expr
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
    basis = defaultdict(int)
    counter = 0
    basis_map = {}  # {frozenset(product_factors): index}

    for expr in exprs:
        terms = expr.args if isinstance(expr, Add) else [expr]
        for term in terms:
            coeff, factors = term.as_coeff_mul()
            # Extract Operator product combinations
            product_factors = tuple(
                sorted(
                    [f for f in factors if isinstance(f, Operator)],
                    key=lambda x: hash(x),
                )
            )
            # Assign basis vector index to each unique product
            key = frozenset(product_factors)
            if key not in basis_map:
                basis_map[key] = counter
                counter += 1
    return list(basis_map.keys())


def build_coefficient_matrix(exprs: list[Expr], basis_map: dict) -> list[list[float]]:
    """Build coefficient matrix (considering product terms)"""
    matrix = []
    for expr in exprs:
        row = [0.0] * len(basis_map)
        terms = expr.args if isinstance(expr, Add) else [expr]
        for term in terms:
            try:
                coeff, factors = term.as_coeff_mul()
                # Extract and sort Operator type factors
                product_factors = tuple(
                    sorted(
                        [f for f in factors if isinstance(f, Operator)],
                        key=lambda x: hash(x),
                    )
                )
                key = frozenset(product_factors)
                if key in basis_map:
                    try:
                        row[basis_map[key]] += float(coeff)
                    except (TypeError, ValueError):
                        # If cannot convert to float, try using 1.0 as default coefficient
                        row[basis_map[key]] += 1.0
            except Exception:
                # If error occurs when processing term, skip it
                continue
        matrix.append(row)
    return matrix


def normalize_with_products(expr: Expr, basis_map: dict) -> Expr:
    """Normalize expression containing product terms, using SymPy symbolic types"""
    try:
        terms = expr.args if isinstance(expr, Add) else [expr]
        coeff_dict = defaultdict(lambda: S.Zero)  # Initialize with SymPy zero

        for term in terms:
            try:
                # Decompose term's coefficient and factors
                coeff, factors = term.as_coeff_mul()
                # Extract and sort Operator type factors
                product_factors = tuple(
                    sorted(
                        (f for f in factors if isinstance(f, Operator)),
                        key=lambda x: hash(x),
                    )
                )
                key = frozenset(product_factors)  # Assume using immutable set as key
                coeff_dict[key] += coeff  # Directly accumulate SymPy coefficients
            except Exception:
                # If error occurs when processing term, skip it
                continue

        # Calculate sum of squares of coefficients
        sum_squares = S.Zero
        for c in coeff_dict.values():
            try:
                sum_squares += c**2
            except Exception:
                # If error in square calculation, try numerical square
                try:
                    sum_squares += float(c) ** 2
                except:
                    # If cannot convert to numerical, use 1 as contribution
                    sum_squares += S.One

        try:
            norm = sqrt(sum_squares)

            # Normalize expression
            if norm == S.Zero:
                return S.Zero  # Avoid division by zero
            return (S.One / norm) * expr
        except Exception:
            # If normalization fails, return original expression
            return expr
    except Exception:
        # If error occurs when processing entire expression, return original expression
        return expr


def find_linear_independent_and_normalized_expr(exprs: list[Expr]) -> list[Expr]:
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
    basis_map = {frozenset(vec): idx for idx, vec in enumerate(basis)}

    # Build coefficient matrix
    coeff_matrix = build_coefficient_matrix(filtered, basis_map)

    # Perform Gaussian elimination directly on original matrix
    mat = Matrix(coeff_matrix)
    _, pivots = mat.rref()

    # Validate pivot elements
    valid_pivots = [i for i in pivots if i < len(filtered)]

    return [normalize_with_products(filtered[i], basis_map) for i in valid_pivots]


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
