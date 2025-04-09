from typing import Literal, NamedTuple, List

import numpy as np
from sympy import Add, Mul, Dummy, Symbol, S, simplify, sqrt
from sympy.physics.quantum import Operator
from .symmetry.sympy_utils import convert_pow_to_mul
from .quark_diagram import Diagram, QuarkDiagram

Flavor = Literal["u", "d", "s", "c", "t", "b"]


class Tag(NamedTuple):
    tag: int
    time: int


class Qurak(Symbol):
    def __new__(cls, flavor: Flavor, tag: Tag, anti: bool, **assumptions) -> None:
        obj = super().__new__(
            cls, Rf"\bar{{{flavor}}}({tag.tag})" if anti else Rf"{flavor}({tag.tag})", commutative=False, **assumptions
        )
        return obj

    def __init__(self, flavor: Flavor, tag: Tag, anti: bool, **assumptions) -> None:
        """
        Initialize a Qurak object.

        Args:
            flavor: The flavor of the quark
            tag: The tag of the quark
            anti: Whether the quark is an anti-quark
            **assumptions: Additional assumptions for the Symbol
        """
        self.flavor = flavor
        self.tag = tag
        self.anti = anti


class Propagator(Symbol):
    """
    Symbol representing quark propagator
    """

    def __new__(cls, flavor: Flavor, source_tag: Tag, sink_tag: Tag, **assumptions) -> None:
        obj = super().__new__(cls, Rf"S^{flavor}({sink_tag.tag}, {source_tag.tag})", **assumptions)
        return obj

    def __init__(self, flavor: Flavor, source_tag: Tag, sink_tag: Tag, **assumptions) -> None:
        """
        Initialize a Propagator object.

        Args:
            flavor: The flavor of the propagator
            source_tag: The source tag of the propagator
            sink_tag: The sink tag of the propagator
            **assumptions: Additional assumptions for the Symbol
        """
        self.flavor = flavor
        self.source_tag = source_tag
        self.sink_tag = sink_tag
        self.tag = Rf"S^{flavor}_\mathrm{{local}}" if source_tag.time == sink_tag.time else Rf"S^{flavor}"


class HadronFlavorStructure(Operator):
    def __new__(cls, flavor_str: str, time: int = 0) -> None:
        if "bar" in flavor_str:
            # 处理形如 bar{uds} 的情况
            obj = super().__new__(cls, rf"bar{{{flavor_str[4:-1]}}}({time})")
        elif len(flavor_str) == 3:
            obj = super().__new__(cls, rf"{flavor_str}({time})")
        elif len(flavor_str) == 2:
            obj = super().__new__(cls, rf"{flavor_str}({time})")
        return obj

    def __init__(self, flavor_str: str, time: int = 0) -> None:
        """
        Initialize a HadronFlavorStructure object.

        Args:
            flavor_str: The flavor string
            time: The time value
        """
        self.flavor_str = flavor_str
        self.time = time

        if "bar" in flavor_str:
            # 处理形如 bar{uds} 的情况
            self.baryon_num = -1
            self.quark_list = []
            self.anti_quark_list = [c for c in flavor_str[4:-1]]
        elif len(flavor_str) == 3:
            self.baryon_num = 1
            self.quark_list = [c for c in flavor_str]
            self.anti_quark_list = []
        elif len(flavor_str) == 2:
            self.baryon_num = 0
            self.quark_list = [flavor_str[1]]
            self.anti_quark_list = [flavor_str[0]]

    def conjugate(self):
        if self.baryon_num == 0:
            new_flavor_str = self.quark_list[0] + self.anti_quark_list[0]
        elif self.baryon_num == 1:
            new_flavor_str = f'bar{{{"".join(self.quark_list)}}}'
        elif self.baryon_num == -1:
            new_flavor_str = "".join(self.anti_quark_list)
        return HadronFlavorStructure(new_flavor_str, self.time)


def quark_contract(expr, particles, degenerate=True):
    """
    Perform quark contraction based on hadron flavor structure in the expression

    Args:
        expr: Expression containing HadronFlavorStructure objects
        particles: List of particles
        degenerate: Whether to consider u and d quark degeneracy

    Returns:
        diagrams: Contraction diagrams
        coeffs: Coefficient list
        particles: Particle list
        propagators: Propagator list
    """
    diagrams = []
    coeffs = []
    propagators = [None]
    expr = convert_pow_to_mul(expr.expand())
    num_particles = len(particles)
    # Expand expression into sum of terms
    terms = Add.make_args(expr)
    result_terms = []
    baryon_num_list = []
    time_list = []
    baryon_num_list_finished = False
    for term in terms:
        # Decompose factors
        factors = Mul.make_args(term)
        coeff = S(1)
        symbol_list = []
        hadron_id = 0
        for factor in factors:
            if isinstance(factor, HadronFlavorStructure):
                if not baryon_num_list_finished:
                    baryon_num_list.append(factor.baryon_num)
                    time_list.append(factor.time)
                # Collect quarks and anti-quarks
                if factor.baryon_num == 0:
                    symbol_list.extend(
                        [
                            Qurak(factor.anti_quark_list[0], Tag(hadron_id * 3, factor.time), True),
                            Qurak(factor.quark_list[0], Tag(hadron_id * 3, factor.time), False),
                        ]
                    )
                elif factor.baryon_num == 1:
                    quark_id = 0
                    for q in factor.quark_list:
                        symbol_list.append(Qurak(q, Tag(hadron_id * 3 + quark_id, factor.time), False))
                        quark_id += 1
                elif factor.baryon_num == -1:
                    quark_id = 0
                    for q in factor.anti_quark_list:
                        symbol_list.append(Qurak(q, Tag(hadron_id * 3 + quark_id, factor.time), True))
                        quark_id += 1
                hadron_id += 1
            else:
                # Non-hadron flavor structure factors as coefficients
                coeff *= factor
        baryon_num_list_finished = True
        # Perform quark contraction
        result_list = []
        result = []

        _quark_contract(symbol_list, result_list, result, degenerate)
        result_terms.append(coeff * Add(*result_list))
    # Merge results and simplify
    terms = Add.make_args(simplify(Add(*result_terms)).expand())

    for term in terms:
        diagram = [[0 for i in range(num_particles)] for j in range(num_particles)]
        for i in range(num_particles):
            for j in range(num_particles):
                if baryon_num_list[i] != 0 and baryon_num_list[j] != 0:
                    diagram[i][j] = [[0 for _ in range(3)] for _ in range(3)]
                elif baryon_num_list[i] != 0 and baryon_num_list[j] == 0:
                    diagram[i][j] = [[0 for _ in range(3)] for _ in range(1)]
                elif baryon_num_list[i] == 0 and baryon_num_list[j] != 0:
                    diagram[i][j] = [[0 for _ in range(1)] for _ in range(3)]
        factors = Mul.make_args(term)
        coeff = S(1)
        for factor in factors:
            if isinstance(factor, Propagator):
                if factor.tag not in propagators:
                    propagators.append(factor.tag)
                hadron_id_source = factor.source_tag.tag // 3
                hadron_id_sink = factor.sink_tag.tag // 3
                quark_id_source = factor.source_tag.tag % 3
                quark_id_sink = factor.sink_tag.tag % 3
                if baryon_num_list[hadron_id_source] == 0 and baryon_num_list[hadron_id_sink] == 0:
                    diagram[hadron_id_source][hadron_id_sink] = propagators.index(factor.tag)
                else:
                    diagram[hadron_id_source][hadron_id_sink][quark_id_source][quark_id_sink] = propagators.index(
                        factor.tag
                    )
            else:
                coeff *= factor

        diagrams.append(diagram)
        coeffs.append(coeff)
    diagram_expr = S(0)
    for i in range(len(diagrams)):
        diagram_expr += coeffs[i] * Diagram(QuarkDiagram(diagrams[i]), time_list, particles, propagators)
    return diagram_expr


def _quark_contract(symbol_list, result_list, result, degenerate):
    if symbol_list == []:
        result_list.append(Mul(*result))
        return
    for i, src in enumerate(symbol_list):
        if src.anti:
            break
    for j, snk in enumerate(symbol_list):
        if not snk.anti and snk.flavor == src.flavor:
            if i > j:
                symbol_list.pop(i)
                symbol_list.pop(j)
                factor = S(-1) ** (i - j - 1)
            else:
                symbol_list.pop(j)
                symbol_list.pop(i)
                factor = S(-1) ** (j - i)
            if degenerate and (snk.flavor == "u" or snk.flavor == "d"):
                prop = Propagator("q", src.tag, snk.tag)
            else:
                prop = Propagator(src.flavor, src.tag, snk.tag)
            result.append(factor * prop)
            _quark_contract(symbol_list, result_list, result, degenerate)
            result.pop()
            if i > j:
                symbol_list.insert(j, snk)
                symbol_list.insert(i, src)
            else:
                symbol_list.insert(i, src)
                symbol_list.insert(j, snk)
