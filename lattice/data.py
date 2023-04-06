from typing import Dict, List

from .insertion import Operator
from .filedata.abstract import FileData


def getElementalData(operators: List[Operator], coefficients: List[float], elemental: FileData):
    from .backend import getBackend
    from .insertion.gamma import gamma

    assert (
        len(operators) == len(coefficients)
    ), F"Unmatched numbers of insertions {len(operators)} and coefficients {len(coefficients)}"
    numpy = getBackend()
    ret = []
    cache: Dict[int, numpy.ndarray] = {}

    for idx in len(operators):
        parts, momentum, coefficient = operators[idx].parts, operators[idx].momentum, coefficients[idx]
        ret_gamma = []
        ret_elemental = []
        for i in range(len(parts) // 2):
            ret_gamma.append(gamma(parts[i * 2]))
            elemental_part = parts[i * 2 + 1]
            for j in range(len(elemental_part)):
                derivative_coeff, derivative = elemental_part[j]
                derivative_coeff *= coefficient
                if derivative not in cache:
                    cache[derivative] = elemental[derivative, momentum]
                if j == 0:
                    ret_elemental.append(derivative_coeff * cache[derivative])
                else:
                    ret_elemental[-1] += derivative_coeff * cache[derivative]
        ret.append((numpy.asarray(ret_gamma), numpy.asarray(ret_elemental)))

    return ret
