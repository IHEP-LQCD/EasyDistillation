import numpy as np
from itertools import permutations
import sympy as sp
from sympy import Matrix
from sympy import Add, Mul, Expr


def antisymmetric_tensor(n):
    """
    Generate an n-th order completely antisymmetric tensor.
    """
    shape = (n,) * n
    tensor = np.zeros(shape, dtype=object)

    # Generate all possible index permutations
    indices = list(permutations(range(n), n))

    for idx in indices:
        sign = 1
        for i in range(n):
            for j in range(i + 1, n):
                if idx[i] > idx[j]:
                    sign *= -1

        tensor[idx] = sign

    return tensor


def generate_hardcoded_code(vals, indent=4):
    def _generate(value, current_indent):
        if isinstance(value, dict):
            # Handle dictionary
            inner_code = "{\n"
            for k, v in value.items():
                inner_code += " " * (current_indent + indent) + f"'{k}': {_generate(v, current_indent + indent)},\n"
            inner_code += " " * current_indent + "}"
            return inner_code
        elif isinstance(value, Matrix):
            # Handle sympy.Matrix
            return f"Matrix({value.tolist()})"
        else:
            # Handle other types (int, float, str, list, etc.)
            return repr(value)

    return f"{_generate(vals, 0)}"


def multiplicationTable(matrix_group):
    keys = list(matrix_group.keys())  # Get string representation of group elements
    n = len(keys)  # Number of matrices
    table = [[None] * n for _ in range(n)]  # Create an n*n table
    # print(keys)
    # Perform matrix multiplication and fill multiplication table
    for i in range(n):
        for j in range(n):
            print(i, j)
            result_matrix = matrix_group[keys[i]] @ matrix_group[keys[j]]  # Calculate matrix multiplication
            # Find the index corresponding to multiplication result
            for k, key in enumerate(keys):
                if abs(((result_matrix - matrix_group[key]).norm()).evalf()) < 0.1:  # Check if equal
                    table[i][j] = k  # Index of result
                    break
            if table[i][j] is None:  # If not found, multiplication result doesn't exist in group
                print(i, j, keys[i], keys[j], result_matrix)
                exit()

    return table


import numpy as np


def are_collinear(arrays):
    """
    Check if multiple arrays are collinear (applicable to NumPy arrays).
    """
    if len(arrays) < 2:
        return True  # Single array is collinear by default

    # Use first array as reference
    base = np.array(arrays[0], dtype=float)
    for arr in arrays[1:]:
        arr = np.array(arr, dtype=float)
        # Calculate cross product norm, if collinear then norm is 0
        cross_product = np.cross(base, arr)
        if not np.allclose(cross_product, 0):
            return False
    return True


def select_nonzero_vector(arrays):
    """
    Select a non-zero vector from collinear arrays.
    """
    for arr in arrays:
        arr = np.array(arr, dtype=float)
        norm = np.linalg.norm(arr)
        if norm > 0:  # Find first non-zero vector
            return arr
    return None  # All vectors are zero vectors


def normalize_array(arr):
    """
    Normalize array.
    """
    norm = np.linalg.norm(arr)
    for i in range(len(arr)):
        if arr[i] != 0:
            sign = np.sign(arr[i])
            break
    return arr / norm * sign  # Preserve direction


def check_and_normalize_arrays(arrays):
    """
    Check if multiple arrays are collinear, normalize if they are, otherwise raise error.
    """
    if not are_collinear(arrays):
        raise ValueError("Arrays are not collinear.")

    # Select non-zero vector
    nonzero_vector = select_nonzero_vector(arrays)
    # Normalize
    if nonzero_vector is None:
        return None  # All vectors are zero vectors, return None
    else:
        normalized_vector = normalize_array(nonzero_vector)
        return normalized_vector


from sympy import simplify, Poly, Expr
from typing import List, Optional


def are_collinear_and_normalize(expressions: List[Expr]) -> Optional[Expr]:
    """
    Check if expression list is collinear, return normalized expression if collinear.

    Parameters:
        expressions: List[Expr] - List of SymPy expressions

    Returns:
        Optional[Expr] - Normalized expression (if collinear), otherwise None
    """
    # Filter non-zero expressions
    non_zero = [expr for expr in expressions if expr != 0]
    if not non_zero:
        return 0  # All expressions are zero

    # Check if all non-zero expressions are collinear
    base = non_zero[0]
    for expr in non_zero[1:]:
        ratio = simplify(expr / base)
        if not ratio.is_constant():
            return None  # Found non-collinear expression

    # Normalize base expression
    try:
        # Try to extract leading coefficient (for polynomials)
        variables = sorted(base.free_symbols, key=str)
        if variables:
            poly = Poly(base, *variables)
            coeff = poly.LC()
        else:
            coeff = base  # Handle constant expressions
        normalized = base / coeff
    except:
        # Non-polynomial expression, try to extract product coefficient
        coeff, terms = base.as_coeff_mul()
        if len(terms) == 1:
            normalized = terms[0]
        else:
            normalized = base  # Cannot decompose further

    return normalized


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
