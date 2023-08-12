from typing import Literal, NamedTuple

from sympy import Add, Mul, Dummy, Symbol, S, simplify, sqrt

Flavor = Literal["u", "d", "s", "c", "t", "b"]


class Tag(NamedTuple):
    tag: int
    time: int


class Insertion(Dummy):
    def __new__(cls, name: str, tag: Tag, dagger: bool, **assumptions) -> None:
        obj = super().__new__(
            cls,
            Rf"{name}^\dagger" if dagger else Rf"{name}",
            commutative=False,
            **assumptions
        )
        obj.tag = tag
        obj.dagger = dagger
        return obj


class Qurak(Symbol):
    def __new__(cls, flavor: Flavor, tag: Tag, anti: bool, **assumptions) -> None:
        obj = super().__new__(
            cls,
            Rf"\bar{{{flavor}}}({tag.tag})" if anti else Rf"{flavor}({tag.tag})",
            commutative=False,
            **assumptions
        )
        obj.flavor = flavor
        obj.tag = tag
        obj.anti = anti
        return obj


class Meson:
    def __init__(
        self, qbar: Flavor, insertion: str, q: Flavor, tag: Tag, dagger: bool = False
    ) -> None:
        if dagger:
            qbar, q = q, qbar
        self.qbar = qbar
        self.q = q
        self.insertion = insertion
        self.tag = tag
        self.dagger = dagger
        self.expression = (
            Qurak(qbar, tag, True)
            * Insertion(insertion, tag, dagger)
            * Qurak(q, tag, False)
        )

    def __add__(self, obj):
        return self.expression + obj

    def __radd__(self, obj):
        return obj + self.expression

    def __mul__(self, obj):
        return self.expression * obj

    def __rmul__(self, obj):
        return obj * self.expression

    def __sub__(self, obj):
        return self.expression + -obj

    def __lsub__(self, obj):
        return obj - self.expression

    def __neg__(self):
        return -self.expression


class Propagator(Symbol):
    def __new__(
        cls, flavor: Flavor, source_tag: Tag, sink_tag: Tag, **assumptions
    ) -> None:
        obj = super().__new__(
            cls, Rf"S^{flavor}({sink_tag.tag}, {source_tag.tag})", **assumptions
        )
        obj.flavor = flavor
        obj.source_tag = source_tag
        obj.sink_tag = sink_tag
        obj.tag = (
            Rf"S^{flavor}_\mathrm{{local}}"
            if source_tag.time == sink_tag.time
            else Rf"S^{flavor}"
        )
        return obj


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


def quark_contract(expr, num_particles, degenerate=True):
    diagrams = []
    coeffs = []
    particles = [None for _ in range(num_particles)]
    propagators = [None]

    terms = Add.make_args(expr.expand())
    result_terms = []
    for term in terms:
        factors = Mul.make_args(term)
        coeff = S(1)
        symbol_list = []
        for factor in factors:
            if not isinstance(factor, Qurak) and not isinstance(factor, Insertion):
                coeff *= factor
            elif isinstance(factor, Qurak):
                symbol_list.append(factor)
            else:
                if particles[factor.tag.tag] is None:
                    particles[factor.tag.tag] = factor.name
        result_list = []
        result = []
        _quark_contract(symbol_list, result_list, result, degenerate)
        result_terms.append(coeff * Add(*result_list))
    terms = Add.make_args(simplify(Add(*result_terms)).expand())
    for term in terms:
        diagram = [[0 for _ in range(num_particles)] for _ in range(num_particles)]
        factors = Mul.make_args(term)
        coeff = S(1)
        for factor in factors:
            if not isinstance(factor, Propagator):
                coeff *= factor
            else:
                if factor.tag not in propagators:
                    propagators.append(factor.tag)
                diagram[factor.source_tag.tag][factor.sink_tag.tag] = propagators.index(
                    factor.tag
                )
        diagrams.append(diagram)
        coeffs.append(coeff)
    return diagrams, coeffs, particles, propagators


a = (
    S(1)
    / sqrt(2)
    * (
        Meson("u", R"γ_5", "u", Tag(0, 0), True)
        + Meson("d", R"γ_5", "d", Tag(0, 0), True)
    )
)
b = (
    S(1)
    / sqrt(2)
    * (
        Meson("u", R"γ_5", "u", Tag(1, 1), False)
        + Meson("d", R"γ_5", "d", Tag(1, 1), False)
    )
)
c = Meson("u", R"γ_5", "d", Tag(0, 0), True)
d = Meson("u", R"γ_5", "d", Tag(1, 1), False)
e = Meson("u", R"γ_i", "u", Tag(0, 0), True)
f = Meson("d", R"γ_5", "u", Tag(1, 1), False)
g = Meson("u", R"γ_5", "d", Tag(2, 1), False)

# print((b * a).expand())
# print((d * c).expand())
# print((g * f * e).expand())
print(quark_contract(b * a, 2))
print(quark_contract(d * c, 2))
print(quark_contract(g * f * e, 3))
