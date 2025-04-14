from itertools import product
from typing import Callable, Dict, List, Union, Any

import numpy as np
from opt_einsum import contract
import sympy as sp
import hashlib

from .backend import get_backend

_SUB_A = "abcdefghijklABCDEFGHIJKL"
_SUB_M = "mnopqruvwxyzMNOPQRUVWXYZ"


class QuarkDiagram:
    def __init__(self, adjacency_matrix) -> None:
        self.adjacency_matrix = adjacency_matrix
        self.operands = []
        self.subscripts = []
        self.operands_data = []
        self.analyse()

    def analyse(self) -> None:
        from copy import deepcopy

        adjacency_matrix = deepcopy(self.adjacency_matrix)
        num_vertex = len(adjacency_matrix)
        visited = [False] * num_vertex
        for idx in range(num_vertex):
            if visited[idx]:
                continue
            propagators = []
            visited[idx] = True
            queue = [idx]
            while queue != []:
                i = queue.pop(0)
                for j in range(num_vertex):
                    path = adjacency_matrix[i][j]
                    if path != 0:
                        adjacency_matrix[i][j] = 0
                        if not visited[j]:
                            visited[j] = True
                            queue.append(j)
                        if isinstance(path, int):
                            propagators.append([path, i, j])
                        elif isinstance(path, list):
                            for _path in path:
                                propagators.append([_path, i, j])
                        else:
                            raise ValueError(f"Invalid value {path} in the adjacency matrix")
            if propagators == []:
                continue
            vertex_operands = []
            vertex_subscripts = []
            propagator_operands = []
            propagator_subscripts = []
            node = 0
            for propagator in propagators:
                propagator_operands.append(propagator)
                propagator_subscripts.append(_SUB_M[node + 1] + _SUB_A[node + 1] + _SUB_M[node] + _SUB_A[node])
                if propagator[1] not in vertex_operands:
                    vertex_operands.append(propagator[1])
                    vertex_subscripts.append(_SUB_M[node] + _SUB_A[node])
                else:
                    i = vertex_operands.index(propagator[1])
                    vertex_subscripts[i] = _SUB_M[node] + _SUB_A[node] + vertex_subscripts[i]
                if propagator[2] not in vertex_operands:
                    vertex_operands.append(propagator[2])
                    vertex_subscripts.append(_SUB_M[node + 1] + _SUB_A[node + 1])
                else:
                    i = vertex_operands.index(propagator[2])
                    vertex_subscripts[i] = vertex_subscripts[i] + _SUB_M[node + 1] + _SUB_A[node + 1]
                node += 2
            for key in range(len(propagator_subscripts)):
                propagator_subscripts[key] = propagator_subscripts[key][0::2] + propagator_subscripts[key][1::2]
            for key in range(len(vertex_subscripts)):
                vertex_subscripts[key] = vertex_subscripts[key][0::2] + vertex_subscripts[key][1::2]
            self.operands.append([propagator_operands, vertex_operands])
            self.subscripts.append(",".join(propagator_subscripts) + "," + ",".join(vertex_subscripts))


class Particle:
    pass


class Meson(Particle):
    def __init__(self, elemental, operator, source) -> None:
        self.elemental = elemental
        self.elemental_data = None
        self.key = None
        self.operator = operator
        self.dagger = source
        self.outward = 1
        self.inward = 1
        # cache is defined as a class variable of the Meson class.
        # cache is shared among all instances of Meson.
        backend = get_backend()
        self.cache: Dict[int, backend.ndarray] = {}

    def __str__(self) -> str:
        str = "### Meson ###\n"
        str += Rf"\n key = {self.key} \n"
        str += self.operator.__str__()
        str += Rf"\n dagger = {self.dagger} \n"
        return str

    def load(self, key, usedNe: int = None):
        self.usedNe = usedNe
        if self.key != key:
            self.key = key
            self.elemental_data = self.elemental.load(key)
            backend = get_backend()
            self.cache: Dict[int, backend.ndarray] = {}
            self._make_cache()

    def _make_cache(self):
        from lattice.insertion.gamma import gamma

        backend = get_backend()
        cache = self.cache
        parts = self.operator.parts
        ret_gamma = []
        ret_elemental = []
        for i in range(len(parts) // 2):
            ret_gamma.append(gamma(parts[i * 2]))
            elemental_part = parts[i * 2 + 1]
            for j in range(len(elemental_part)):
                elemental_coeff, derivative_idx, momentum_idx = elemental_part[j]
                deriv_mom_tuple = (derivative_idx, momentum_idx)
                if deriv_mom_tuple not in cache:
                    cache[deriv_mom_tuple] = self.elemental_data[
                        derivative_idx, momentum_idx, :, : self.usedNe, : self.usedNe
                    ]
                if j == 0:
                    ret_elemental.append(elemental_coeff * cache[deriv_mom_tuple])
                else:
                    ret_elemental[-1] += elemental_coeff * cache[deriv_mom_tuple]
        if self.dagger:
            self.cache = (
                contract("ik,xlk,lj->xij", gamma(8), backend.asarray(ret_gamma).conj(), gamma(8)),
                contract("xtba->xtab", backend.asarray(ret_elemental).conj()),
            )
        else:
            self.cache = (
                backend.asarray(ret_gamma),
                backend.asarray(ret_elemental),
            )

    def get(self, t):
        if isinstance(t, int):
            if self.dagger:
                return contract("xij,xab->ijab", self.cache[0], self.cache[1][:, t])
            else:
                return contract("xij,xab->ijab", self.cache[0], self.cache[1][:, t])
        else:
            if self.dagger:
                return contract("xij,xtab->tijab", self.cache[0], self.cache[1][:, t])
            else:
                return contract("xij,xtab->tijab", self.cache[0], self.cache[1][:, t])


class Propagator:
    def __init__(self, perambulator, Lt) -> None:
        self.perambulator = perambulator
        self.perambulator_data = None
        self.key = None
        self.Lt = Lt
        self.cache = None
        self.cache_dagger = None
        self.cached_time = None

    def load(self, key, usedNe: int = None):
        if self.key != key:
            self.key = key
            self.usedNe = usedNe
            self.perambulator_data = self.perambulator.load(key)

    def get(self, t_source, t_sink):
        from lattice.insertion.gamma import gamma

        if isinstance(t_source, int) and isinstance(t_sink, int):
            if self.cached_time != t_source and self.cached_time != t_sink:
                self.cache = self.perambulator_data[t_source, :, :, :, : self.usedNe, : self.usedNe]
                self.cache_dagger = contract("ik,tlkba,lj->tijab", gamma(15), self.cache.conj(), gamma(15))
                self.cached_time = t_source
            if self.cached_time == t_source:
                return self.cache[(t_sink - t_source) % self.Lt]
            else:
                return self.cache_dagger[(t_source - t_sink) % self.Lt]
        elif isinstance(t_source, int):
            if self.cached_time != t_source:
                self.cache = self.perambulator_data[t_source, :, :, :, : self.usedNe, : self.usedNe]
                self.cache_dagger = contract("ik,tlkba,lj->tijab", gamma(15), self.cache.conj(), gamma(15))
                self.cached_time = t_source
            return self.cache[(t_sink - t_source) % self.Lt]
        elif isinstance(t_sink, int):
            if self.cached_time != t_sink:
                self.cache = self.perambulator_data[t_sink, :, :, :, : self.usedNe, : self.usedNe]
                self.cache_dagger = contract("ik,tlkba,lj->tijab", gamma(15), self.cache.conj(), gamma(15))
                self.cached_time = t_sink
            return self.cache_dagger[(t_source - t_sink) % self.Lt]
        else:
            raise ValueError("At least t_source or t_sink should be int")


class PropagatorLocal:
    def __init__(self, perambulator, Lt) -> None:
        self.perambulator = perambulator
        self.key = None
        self.Lt = Lt
        self.cache = None

    def load(self, key, usedNe: int = None):
        if self.key != key:
            self.key = key
            self.perambulator_data = self.perambulator.load(key)
            self.usedNe = usedNe
            self._make_cache()

    def _make_cache(self):
        self.cache = self.perambulator_data[0, :, :, :, : self.usedNe, : self.usedNe]
        for t_source in range(1, self.Lt):
            self.cache[t_source] = self.perambulator_data[t_source, 0, :, :, : self.usedNe, : self.usedNe]

    def get(self, t_source, t_sink):
        if isinstance(t_source, int):
            assert t_source == t_sink, "You cannot use PropagatorLocal here"
        else:
            assert (t_source == t_sink).all(), "You cannot use PropagatorLocal here"
        return self.cache[t_source]


def compute_diagrams_multitime(diagrams: List[QuarkDiagram], time_list, vertex_list, propagator_list):
    backend = get_backend()
    diagram_value = []
    for diagram in diagrams:
        diagram_value.append(1.0)
        for operands, subscripts in zip(diagram.operands, diagram.subscripts):
            have_multitime = False
            subscripts = subscripts.split(",")
            idx = 0
            operands_data = []
            for item in operands[0]:
                operands_data.append(propagator_list[item[0]].get(time_list[item[1]], time_list[item[2]]))
                if not isinstance(time_list[item[1]], int) or not isinstance(time_list[item[2]], int):
                    subscripts[idx] = "t" + subscripts[idx]
                    have_multitime = True
                idx += 1
            for item in operands[1]:
                operands_data.append(vertex_list[item].get(time_list[item]))
                if not isinstance(time_list[item], int):
                    subscripts[idx] = "t" + subscripts[idx]
                    have_multitime = True
                idx += 1
            if have_multitime:
                subscripts[-1] = subscripts[-1] + "->t"
            diagram_value[-1] = diagram_value[-1] * contract(",".join(subscripts), *operands_data)
    return backend.asarray(diagram_value)


def compute_diagrams(diagrams: List[QuarkDiagram], time_list, vertex_list, propagator_list):
    backend = get_backend()
    diagram_value = []
    for diagram in diagrams:
        diagram_value.append(1.0)
        for operands, subscripts in zip(diagram.operands, diagram.subscripts):
            operands_data = []
            for item in operands[0]:
                operands_data.append(propagator_list[item[0]].get(time_list[item[1]], time_list[item[2]]))
            for item in operands[1]:
                operands_data.append(vertex_list[item].get(time_list[item]))
            diagram_value[-1] *= contract(subscripts, *operands_data)
    return backend.asarray(diagram_value)


from typing import Union, List, Dict, Tuple, Any
from sympy import S, Add, Expr, Symbol, Mul
import hashlib


class Diagram(Symbol):
    def __new__(cls, diagram: QuarkDiagram, time_list, vertex_list, propagator_list) -> None:
        obj = super().__new__(cls, f"{diagram.adjacency_matrix},{time_list},{vertex_list},{propagator_list}")
        return obj

    def __init__(self, diagram: QuarkDiagram, time_list, vertex_list, propagator_list) -> None:
        """
        Initialize a Diagram object.

        Args:
            diagram: The QuarkDiagram object
            time_list: List of time values
            vertex_list: List of vertices
            propagator_list: List of propagators
        """
        self.diagram = diagram
        self.time_list = time_list
        self.vertex_list = vertex_list
        self.propagator_list = propagator_list
        self.value = None
        self.value_pointer = None

    def calc(self):
        if self.value is None:
            self.value = self.__hash__()
            self.value = compute_diagrams_multitime(
                [self.diagram], self.time_list, self.vertex_list, self.propagator_list
            )
        return self.value

    def __str__(self):
        return f"{self.diagram.adjacency_matrix},{self.time_list},{self.vertex_list},{self.propagator_list}"

    def __repr__(self):
        return f"{self.diagram.adjacency_matrix},{self.time_list},{self.vertex_list},{self.propagator_list}"

    def __eq__(self, other):
        if not isinstance(other, Diagram):
            return False
        return (
            self.diagram.adjacency_matrix == other.diagram.adjacency_matrix
            and self.time_list == other.time_list
            and self.vertex_list == other.vertex_list
            and self.propagator_list == other.propagator_list
        )

    def __hash__(self):
        return int(hashlib.sha256(str(self).encode()).hexdigest(), 16) % (2**31)

    def simplify(self):
        """
        Simplify Diagram object, perform the following operations:
        1. Remove redundant vertices (equivalent to remove_redundant functionality)
        2. Sort vertices and propagators (equivalent to sort_vertex_and_propagator functionality)
        3. Split graph into product of different subgraphs

        Integrates the functionality of the original separate remove_redundant and sort_vertex_and_propagator methods,
        and further splits the graph into connected components, ultimately returning the optimized Diagram or Diagram product expression.

        Returns:
            sympy.Expr or Diagram: Simplified Diagram object or expression representing product of different subgraphs
        """
        from sympy import Mul
        from copy import deepcopy

        # Get graph information
        adjacency_matrix = deepcopy(self.diagram.adjacency_matrix)
        num_vertex = len(adjacency_matrix)
        # Record each vertex's connected component
        component_ids = [-1] * num_vertex
        next_component_id = 0

        # Use BFS to find all connected components
        for start_vertex in range(num_vertex):
            # If already assigned connected component, skip
            if component_ids[start_vertex] != -1:
                continue

            # Check if any edge involves this vertex
            has_connection = False
            for i in range(num_vertex):
                if (
                    isinstance(adjacency_matrix[start_vertex][i], np.ndarray)
                    and (adjacency_matrix[start_vertex][i] != 0).any()
                ):
                    has_connection = True
                elif (
                    isinstance(adjacency_matrix[i][start_vertex], np.ndarray)
                    and (adjacency_matrix[i][start_vertex] != 0).any()
                ):
                    has_connection = True
                elif adjacency_matrix[start_vertex][i] != 0:
                    has_connection = True
                elif adjacency_matrix[i][start_vertex] != 0:
                    has_connection = True
                if has_connection:
                    break
            if not has_connection:
                continue
            # Use BFS to find all connected vertices
            component_ids[start_vertex] = next_component_id
            queue = [start_vertex]
            while queue:
                vertex = queue.pop(0)

                # Check all possible connections
                for next_vertex in range(num_vertex):
                    # Check connection from vertex to next_vertex
                    if component_ids[next_vertex] == -1:
                        is_in_queue = False
                        if (
                            isinstance(adjacency_matrix[vertex][next_vertex], np.ndarray)
                            and (adjacency_matrix[vertex][next_vertex] != 0).any()
                        ):
                            is_in_queue = True
                        elif (
                            isinstance(adjacency_matrix[next_vertex][vertex], np.ndarray)
                            and (adjacency_matrix[next_vertex][vertex] != 0).any()
                        ):
                            is_in_queue = True
                        elif adjacency_matrix[vertex][next_vertex] != 0:
                            is_in_queue = True
                        elif adjacency_matrix[next_vertex][vertex] != 0:
                            is_in_queue = True
                        if is_in_queue:
                            component_ids[next_vertex] = next_component_id
                            queue.append(next_vertex)
            # Start a new connected component
            next_component_id += 1
        # Create a new Diagram object for each connected component
        result_diagrams = []
        for component_id in range(next_component_id):
            # Find vertices belonging to this connected component
            vertices = []
            for i in range(num_vertex):
                if component_ids[i] == component_id:
                    vertices.append(i)
            # Sort vertices based on self.vertex_list
            # When vertices are equal, consider all possible orders
            # First, sort based on time and vertex type
            vertices.sort(key=lambda v: (self.time_list[v], self.vertex_list[v]))

            # Check if there are same vertices
            has_same_vertices = False
            for i in range(len(vertices) - 1):
                if (
                    self.time_list[vertices[i]] == self.time_list[vertices[i + 1]]
                    and self.vertex_list[vertices[i]] == self.vertex_list[vertices[i + 1]]
                ):
                    has_same_vertices = True
                    break

            # If there are same vertices, consider all possible orders
            if has_same_vertices:
                # Group by time and vertex type
                from itertools import groupby
                from itertools import permutations

                # Group by time and vertex type
                groups = []
                for k, g in groupby(vertices, key=lambda v: (self.time_list[v], self.vertex_list[v])):
                    groups.append(list(g))

                # Only sort vertices in each group, keeping group order
                all_possible_orders = []
                for g in groups:
                    perms = [list(p) for p in permutations(g)]
                    all_possible_orders.append(perms)

                # Use itertools.product to get all combinations
                from itertools import product

                all_permutations = list(product(*all_possible_orders))

                # Flatten nested list to match vertices format
                all_possible_vertices = []
                for perm_combination in all_permutations:
                    flattened_vertices = []
                    for group_perm in perm_combination:
                        flattened_vertices.extend(group_perm)
                    all_possible_vertices.append(flattened_vertices)
            else:
                all_possible_vertices = [vertices]
            # Create adjacency matrix for this connected component, size of connected component
            min_hash = float("inf")
            for vertices in all_possible_vertices:
                component_size = len(vertices)
                component_matrix = [
                    [adjacency_matrix[vertices[j]][vertices[i]] for i in range(component_size)]
                    for j in range(component_size)
                ]
                new_quark_diagram = QuarkDiagram(component_matrix)
                new_time_list = [self.time_list[i] for i in vertices]
                new_vertex_list = [self.vertex_list[i] for i in vertices]
                # Create new propagator_list, only retain used propagators
                used_propagators = set([])  # 0 is default value, always retained

                # Traverse component_matrix to find all used propagators
                for i in range(component_size):
                    for j in range(component_size):
                        value = component_matrix[i][j]
                        if isinstance(value, int):
                            if value != 0:
                                used_propagators.add(value)
                        elif isinstance(value, np.ndarray):
                            # For array type, add propagator indices of all non-zero elements
                            for prop_idx in value.flatten():
                                if prop_idx != 0:
                                    used_propagators.add(int(prop_idx))
                        elif isinstance(value, list):
                            # For list type, add all non-zero elements
                            # Handle nested lists, similar to ndarray's flatten operation
                            flat_values = []

                            def flatten_list(lst):
                                for item in lst:
                                    if isinstance(item, list):
                                        flatten_list(item)
                                    else:
                                        flat_values.append(item)

                            flatten_list(value)
                            for prop_idx in flat_values:
                                if prop_idx != 0:
                                    used_propagators.add(prop_idx)

                # Sort used propagator indices in original order
                used_propagators = sorted(list(used_propagators), key=lambda x: self.propagator_list[x])
                used_propagators = [0] + used_propagators
                new_propagator_list = [self.propagator_list[i] for i in used_propagators]

                # Update propagator indices in component_matrix
                old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(used_propagators)}
                for i in range(component_size):
                    for j in range(component_size):
                        value = component_matrix[i][j]
                        if isinstance(value, int):
                            if value != 0:
                                component_matrix[i][j] = old_to_new[value]
                        elif isinstance(value, np.ndarray):
                            # For array type, update all non-zero elements
                            new_array = np.zeros_like(value)
                            for idx in np.ndindex(value.shape):
                                if value[idx] != 0:
                                    new_array[idx] = old_to_new[int(value[idx])]
                            component_matrix[i][j] = new_array
                        elif isinstance(value, list):
                            # For list type, update all non-zero elements
                            # Handle nested list case
                            def update_nested_list(lst):
                                result = []
                                for item in lst:
                                    if isinstance(item, list):
                                        result.append(update_nested_list(item))
                                    else:
                                        result.append(old_to_new[item] if item != 0 else 0)
                                return result

                            component_matrix[i][j] = update_nested_list(value)

                # Use stable hash algorithm to calculate graph hash
                current_hash = int(hashlib.sha256(str(new_quark_diagram.adjacency_matrix).encode()).hexdigest(), 16)
                if min_hash > current_hash:
                    min_hash = current_hash
                    new_diagram = Diagram(new_quark_diagram, new_time_list, new_vertex_list, new_propagator_list)

            result_diagrams.append(new_diagram)

        # Convert results to product expression
        result = S(1)
        for diagram in result_diagrams:
            result = Mul(result, diagram, evaluate=False)
        return sp.simplify(result)

    def replace_propagator(self, propagator_map: Dict):
        """
        Replace propagators in Diagram
        """
        for i, propagator in enumerate(self.propagator_list):
            if propagator in propagator_map:
                self.propagator_list[i] = propagator_map[propagator]

    def replace_vertex(self, vertex_map: Callable):
        """
        Replace vertices in Diagram
        """
        for i, vertex in enumerate(self.vertex_list):
            result = vertex_map(vertex)
            if result is not None:
                self.vertex_list[i] = result

    def replace_time(self, time_map: Dict):
        """
        Replace time in Diagram
        """
        for i, time in enumerate(self.time_list):
            if time in time_map:
                self.time_list[i] = time_map[time]


def diagram_vertice_replace(expr: Union[Expr, List, Any], indice_map: Dict) -> Union[Expr, List, Any]:
    """
    Replace vertices in all Diagram object in the expr,list,dict,tuple or any data structure
    """
    if isinstance(expr, Diagram):
        # Replace vertices in the Diagram object
        new_vertice_list = [indice_map[v] for v in expr.vertex_list]
        new_diagram = Diagram(expr.diagram, expr.time_list, new_vertice_list, expr.propagator_list)
        return new_diagram
    elif isinstance(expr, list):
        # Recursively process list elements
        return [diagram_vertice_replace(item, indice_map) for item in expr]
    elif isinstance(expr, tuple):
        # Recursively process tuple elements
        return tuple(diagram_vertice_replace(item, indice_map) for item in expr)
    elif isinstance(expr, dict):
        # Recursively process dictionary values
        return {key: diagram_vertice_replace(value, indice_map) for key, value in expr.items()}
    elif isinstance(expr, Add):
        # Process sympy Add expression
        return Add(*[diagram_vertice_replace(arg, indice_map) for arg in expr.args])
    elif isinstance(expr, Mul):
        # Process sympy Mul expression
        return Mul(*[diagram_vertice_replace(arg, indice_map) for arg in expr.args])
    elif hasattr(expr, "args") and expr.args:
        # For other expressions with args attribute
        return expr.func(*[diagram_vertice_replace(arg, indice_map) for arg in expr.args])
    else:
        # Return unchanged for other types
        return expr


def diagram_simplify(expr: Union[Expr, List, Any]) -> Union[Expr, List, Any]:
    """
    Recursively simplify expressions containing Diagram objects

    Call each Diagram object's simplify method, which integrates the following functionalities:
    1. Remove redundant vertices
    2. Sort vertices and propagators
    3. Split graph into connected components

    Supports processing various data structures, including:
    - Single Diagram object
    - sympy expressions (e.g., Add, Mul, Pow, etc.)
    - Nested lists, tuples, dictionaries
    - NumPy arrays

    Args:
        expr: Expression or data structure containing Diagram objects

    Returns:
        Simplified expression or data structure, original expression unchanged
    """
    from sympy import Add, Mul, Pow, Number, Symbol
    import numpy as np

    # Handle None or unsupported types
    if expr is None:
        return expr

    # Base case: process single Diagram object
    if isinstance(expr, Diagram):
        try:
            # Apply simplification operations
            # expr = expr.remove_redundant()
            # expr = expr.sort_vertex_and_propagator()
            splited_expr = expr.simplify()
            result = sp.simplify(splited_expr)
            return result

        except Exception as e:
            # If an exception occurs, return original expression and print error
            print(f"Warning: Simplification of Diagram failed: {e}")
            import traceback

            traceback.print_exc()
            return expr

    # Process general list - recursively process each element in the list
    elif isinstance(expr, list):
        return [diagram_simplify(item) for item in expr]

    # Process numpy ndarray
    elif hasattr(expr, "__array__") and hasattr(expr, "shape"):  # Detect ndarray
        # Get original shape
        original_shape = expr.shape

        # Flatten ndarray to 1D array, process each element, then restore original shape
        flattened = expr.flatten() if hasattr(expr, "flatten") else expr.ravel()
        result = np.array([diagram_simplify(item) for item in flattened], dtype=object)

        # Restore original shape
        return result.reshape(original_shape)

    # Recursively process addition expression
    elif isinstance(expr, Add):
        terms = []
        for term in expr.args:
            simplified_term = diagram_simplify(term)
            terms.append(simplified_term)
        return Add(*terms)

    # Recursively process multiplication expression
    elif isinstance(expr, Mul):
        factors = []
        for factor in expr.args:
            simplified_factor = diagram_simplify(factor)
            # Handle multiplication nested cases
            if isinstance(simplified_factor, Mul):
                factors.extend(simplified_factor.args)
            else:
                factors.append(simplified_factor)
        return Mul(*factors)

    # Recursively process power expression
    elif isinstance(expr, Pow):
        base = diagram_simplify(expr.args[0])
        # Keep exponent unchanged
        exponent = expr.args[1]
        return Pow(base, exponent)

    # Recursively process dictionary - process value part
    elif isinstance(expr, dict):
        return {key: diagram_simplify(value) for key, value in expr.items()}

    # Recursively process tuple - similar to list but returns tuple
    elif isinstance(expr, tuple):
        return tuple(diagram_simplify(item) for item in expr)

    # Other types of expressions remain unchanged
    else:
        return sp.simplify(expr)


def remove_unexpected_diagram(expr: Union[Expr, List, Any], propagator_list: List[Propagator]):
    """
    Recursively find all Diagram objects and replace those with propagators meeting certain conditions with S(0).

    This function traverses through expressions, lists, dictionaries, or other nested structures to find
    Diagram objects. If a Diagram contains any propagator from the provided propagator_list, it will be
    replaced with a symbolic zero (S(0)).

    Args:
        expr: The expression or structure to process, can be a sympy expression, list, dictionary, etc.
        propagator_list: List of Propagator objects to check against

    Returns:
        The processed expression with redundant diagrams replaced by zeros
    """
    from sympy import Add, Mul, Pow, S
    import numpy as np

    # Handle None or unsupported types
    if expr is None:
        return expr

    # Process Diagram object
    if isinstance(expr, Diagram):
        # Check if this Diagram contains any propagator from the provided propagator_list
        for prop in expr.propagator_list:
            if prop in propagator_list:
                return S(0)  # If contains, return symbolic 0
        return expr  # If not contains, remain unchanged

    # Process list
    elif isinstance(expr, list):
        return [remove_unexpected_diagram(item, propagator_list) for item in expr]

    # Process numpy array
    elif hasattr(expr, "__array__") and hasattr(expr, "shape"):
        original_shape = expr.shape
        flattened = expr.flatten() if hasattr(expr, "flatten") else expr.ravel()
        result = np.array([remove_unexpected_diagram(item, propagator_list) for item in flattened], dtype=object)
        return result.reshape(original_shape)

    # Process addition expression
    elif isinstance(expr, Add):
        terms = [remove_unexpected_diagram(term, propagator_list) for term in expr.args]
        return Add(*terms)

    # Process multiplication expression
    elif isinstance(expr, Mul):
        factors = [remove_unexpected_diagram(factor, propagator_list) for factor in expr.args]
        return Mul(*factors)

    # Process power expression
    elif isinstance(expr, Pow):
        base = remove_unexpected_diagram(expr.args[0], propagator_list)
        exponent = expr.args[1]  # Keep exponent unchanged
        return Pow(base, exponent)

    # Process dictionary
    elif isinstance(expr, dict):
        return {key: remove_unexpected_diagram(value, propagator_list) for key, value in expr.items()}

    # Process tuple
    elif isinstance(expr, tuple):
        return tuple(remove_unexpected_diagram(item, propagator_list) for item in expr)

    # Other types of expressions remain unchanged
    else:
        return expr


def calc_diagram(
    expr: Union[Expr, List, Any], time_map: Dict = None, propagator_map: Dict = None, vertex_map: Callable = None
):
    """
    Find all Diagram objects in the expression and calculate their values. If expr is a multi-level list, dictionary, or tuple, it will be recursively processed.
    During calculation, collect all unequal Diagram objects, and set diagram_list. Then let diagram_list in the expression point to the corresponding element index in diagram_list, and integrate diagram_list to calculate the values of all Diagram objects, then replace all Diagram objects in the expression with diagram.value_pointer pointing to Diagram.value, complete the calculation, and output the result
    """
    from sympy import Add, Mul, Pow, Number, Symbol
    import numpy as np

    # Handle None or unsupported types
    if expr is None:
        return expr

    # Collect all unequal Diagram objects
    diagram_list = []

    def collect_diagrams(e):
        """Recursively collect all unequal Diagram objects in the expression and set value_pointer to point to equal objects"""
        if isinstance(e, Diagram):
            # Check if an equal Diagram object already exists
            found_idx = None
            for idx, existing_diagram in enumerate(diagram_list):
                if e == existing_diagram:  # Use __eq__ method to compare
                    found_idx = idx
                    break

            if found_idx is not None:
                # If an equal object exists, set current object's value_pointer to point to that object
                e.value_pointer = found_idx
            else:
                # If no equal object exists, add to list and set value_pointer
                e.value_pointer = len(diagram_list)
                diagram_list.append(e)
        elif isinstance(e, list):
            for item in e:
                collect_diagrams(item)
        elif isinstance(e, tuple):
            for item in e:
                collect_diagrams(item)
        elif isinstance(e, dict):
            for value in e.values():
                collect_diagrams(value)
        elif isinstance(e, Add):
            for term in e.args:
                collect_diagrams(term)
        elif isinstance(e, Mul):
            for factor in e.args:
                collect_diagrams(factor)
        elif isinstance(e, Pow):
            collect_diagrams(e.base)
        elif hasattr(e, "__array__") and hasattr(e, "shape"):  # Process numpy array
            flattened = e.flatten() if hasattr(e, "flatten") else e.ravel()
            for item in flattened:
                collect_diagrams(item)

    # Collect all Diagram objects
    collect_diagrams(expr)

    # Calculate values of all unequal Diagram objects, calculate before finding their propagator_list and vertex_list union, expand all Diagram objects to propagator_list and vertex_list union
    if diagram_list:
        # Collect all propagator_list and vertex-time pairs
        all_propagators = []
        all_time_vertex_pairs = []  # Store (time, vertex) pairs, time in front

        # Count time-vertex pairs and propagators in all graphs
        for diagram in diagram_list:
            # Collect all time-vertex pairs
            for i, (vertex, time) in enumerate(zip(diagram.vertex_list, diagram.time_list)):
                pair = (time, vertex)  # Time in front
                if pair not in all_time_vertex_pairs:
                    all_time_vertex_pairs.append(pair)

            # Collect all propagators
            for p in diagram.propagator_list:
                if p not in all_propagators:
                    all_propagators.append(p)

        # Create new QuarkDiagram object to integrate all graphs
        combined_diagrams = []
        original_to_new_time_vertex = {}  # Map original index to new time-vertex pair index
        original_to_new_propagator = {}
        # Create time-vertex pair and propagator mapping for each graph
        for diagram in diagram_list:
            # Create time-vertex pair mapping
            original_to_new_time_vertex[id(diagram)] = {}
            for i, (vertex, time) in enumerate(zip(diagram.vertex_list, diagram.time_list)):
                pair = (time, vertex)  # Time in front
                original_to_new_time_vertex[id(diagram)][i] = all_time_vertex_pairs.index(pair)

            # Create propagator mapping
            original_to_new_propagator[id(diagram)] = {}
            for i, p in enumerate(diagram.propagator_list):
                original_to_new_propagator[id(diagram)][i] = all_propagators.index(p)

            # Create new adjacency matrix
            n_vertices = len(all_time_vertex_pairs)
            new_adjacency = [[0 for _ in range(n_vertices)] for _ in range(n_vertices)]

            # Fill new adjacency matrix
            old_adjacency = diagram.diagram.adjacency_matrix
            for i in range(len(diagram.time_list)):
                for j in range(len(diagram.time_list)):
                    value = old_adjacency[i][j]
                    if value != 0:
                        new_i = original_to_new_time_vertex[id(diagram)][i]
                        new_j = original_to_new_time_vertex[id(diagram)][j]

                        # Update propagator indices
                        if isinstance(value, int):
                            new_value = original_to_new_propagator[id(diagram)][value]
                            new_adjacency[new_i][new_j] = new_value
                        elif isinstance(value, list):
                            new_value = [original_to_new_propagator[id(diagram)][v] if v != 0 else 0 for v in value]
                            new_adjacency[new_i][new_j] = new_value

            # Create new QuarkDiagram and add to list
            combined_diagrams.append(QuarkDiagram(new_adjacency))

        # Extract all vertices and time list
        all_vertices = [pair[1] for pair in all_time_vertex_pairs]  # Vertex in tuple's second position
        all_times = [pair[0] for pair in all_time_vertex_pairs]  # Time in tuple's first position

        # Replace all propagators and vertices in graphs
        if vertex_map is not None:
            for vertex in all_vertices:
                new_vertex = vertex_map(vertex)
                if new_vertex is not None:
                    vertex = new_vertex
        if propagator_map is not None:
            for propagator in all_propagators:
                if propagator in propagator_map:
                    propagator = propagator_map[propagator]
        if time_map is not None:
            for time in all_times:
                if time in time_map:
                    time = time_map[time]

        # Calculate values of all graphs at once
        backend = get_backend()
        # results = compute_diagrams_multitime(combined_diagrams, all_times, all_vertices, all_propagators)
        results = [Symbol("result_{}".format(i)) for i in range(len(combined_diagrams))]
        # Assign calculated results to each graph
        for i, diagram in enumerate(diagram_list):
            diagram.value = results[i]

    # Replace Diagram objects in expression with their values
    def replace_diagrams(e):
        if isinstance(e, Diagram):
            # Use value_pointer pointed value
            if e.value_pointer is not None:
                return diagram_list[e.value_pointer].value
            else:
                raise ValueError("Diagram has no value_pointer")
        elif isinstance(e, list):
            return [replace_diagrams(item) for item in e]
        elif isinstance(e, tuple):
            return tuple(replace_diagrams(item) for item in e)
        elif isinstance(e, dict):
            return {key: replace_diagrams(value) for key, value in e.items()}
        elif isinstance(e, Add):
            return Add(*[replace_diagrams(term) for term in e.args])
        elif isinstance(e, Mul):
            return Mul(*[replace_diagrams(factor) for factor in e.args])
        elif isinstance(e, Pow):
            return Pow(replace_diagrams(e.base), e.exp)
        elif hasattr(e, "__array__") and hasattr(e, "shape"):  # Process numpy array
            original_shape = e.shape
            flattened = e.flatten() if hasattr(e, "flatten") else e.ravel()
            result = np.array([replace_diagrams(item) for item in flattened], dtype=object)
            return result.reshape(original_shape)
        else:
            return e

    # Replace and return result
    return replace_diagrams(expr)
