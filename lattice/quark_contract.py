from sympy import Symbol, sqrt, S, Add, Mul, simplify


class Insertion(Symbol):
    def __new__(cls, name, dagger, **assumptions) -> None:
        obj = super().__new__(
            cls,
            Rf"{name}^\dagger" if dagger else Rf"{name}",
            commutative=False,
            **assumptions
        )
        return obj


class Qurak(Symbol):
    def __new__(cls, flavor, tag, anti, **assumptions) -> None:
        obj = super().__new__(
            cls,
            Rf"\bar{{{flavor}}}({tag})" if anti else Rf"{flavor}({tag})",
            commutative=False,
            **assumptions
        )
        obj.flavor = flavor
        obj.tag = tag
        obj.anti = anti
        return obj


class Meson:
    def __init__(
        self, qbar: str, insertion: str, q: str, tag: str, dagger: bool
    ) -> None:
        if dagger:
            self.qbar, self.q = q, qbar
        else:
            self.qbar, self.q = qbar, q
        self.insertion = insertion
        self.tag = tag
        self.dagger = dagger
        self.expression = (
            Qurak(self.qbar, tag, True)
            * Insertion(insertion, dagger)
            * Qurak(self.q, tag, False)
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
    def __new__(cls, flavor, source_tag, sink_tag, **assumptions) -> None:
        obj = super().__new__(
            cls, Rf"S^{flavor}({sink_tag}, {source_tag})", **assumptions
        )
        obj.flavor = flavor
        obj.source_tag = source_tag
        obj.sink_tag = sink_tag
        return obj


def _quark_contract(symbol_list, result_list, temp, degenerate):
    if symbol_list == []:
        result_list.append(Mul(*temp))
        temp.clear()
        return
    for i, snk in enumerate(symbol_list):
        if snk.anti:
            break
    for j, src in enumerate(symbol_list):
        if not src.anti and src.flavor == snk.flavor:
            if i > j:
                symbol_list.pop(i)
                symbol_list.pop(j)
                factor = S(-1) ** (i - j - 1)
            else:
                symbol_list.pop(j)
                symbol_list.pop(i)
                factor = S(-1) ** (j - i)
            if degenerate and (src.flavor == "u" or src.flavor == "d"):
                prop = Propagator("q", src.tag, snk.tag)
            else:
                prop = Propagator(src.flavor, src.tag, snk.tag)
            temp.append(factor * prop)
            _quark_contract(symbol_list, result_list, temp, degenerate)
            if i > j:
                symbol_list.insert(j, src)
                symbol_list.insert(i, snk)
            else:
                symbol_list.insert(i, snk)
                symbol_list.insert(j, src)


def quark_contract(expr, degenerate=True):
    terms = Add.make_args(expr.expand())
    result_term = []
    for term in terms:
        factors = Mul.make_args(term)
        coeff = S(1)
        symbol_list = []
        for factor in factors:
            if not isinstance(factor, Qurak) and not isinstance(factor, Insertion):
                coeff *= factor
            elif isinstance(factor, Qurak):
                symbol_list.append(factor)
        result_list = []
        temp = []
        _quark_contract(symbol_list, result_list, temp, degenerate)
        result_term.append(coeff * Add(*result_list))
    return simplify(Add(*result_term))


a = (
    S(1)
    / sqrt(2)
    * (
        Meson("u", R"\gamma_5", "u", "0", False)
        + Meson("d", R"\gamma_5", "d", "0", False)
    )
)
b = (
    S(1)
    / sqrt(2)
    * (
        Meson("u", R"\gamma_5", "u", "1", True)
        + Meson("d", R"\gamma_5", "d", "1", True)
    )
)
c = Meson("u", R"\gamma_5", "d", "0", False)
d = Meson("u", R"\gamma_5", "d", "1", True)

print(quark_contract(a * b))
print(quark_contract(c * d))
