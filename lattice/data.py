from typing import Dict, List

from .insertion import Operator
from .filedata.abstract import FileData


def get_elemental_data(operators: List[Operator], elemental: FileData, usedNe: int):
    from .backend import get_backend
    from .insertion.gamma import gamma

    backend = get_backend()
    ret = []
    cache: Dict[int, backend.ndarray] = {}

    for operator in operators:
        parts = operator.parts
        ret_gamma = []
        ret_elemental = []
        for i in range(len(parts) // 2):
            ret_gamma.append(gamma(parts[i * 2]))
            elemental_part = parts[i * 2 + 1]
            for j in range(len(elemental_part)):
                elemental_coeff, derivative_idx, momentum_idx = elemental_part[j]
                deriv_mom_tuple = (derivative_idx, momentum_idx)
                if deriv_mom_tuple not in cache:
                    cache[deriv_mom_tuple] = elemental[derivative_idx, momentum_idx, :, :usedNe, :usedNe]
                if j == 0:
                    ret_elemental.append(elemental_coeff * cache[deriv_mom_tuple])
                else:
                    ret_elemental[-1] += elemental_coeff * cache[deriv_mom_tuple]
        ret.append((backend.asarray(ret_gamma), backend.asarray(ret_elemental)))

    return ret
