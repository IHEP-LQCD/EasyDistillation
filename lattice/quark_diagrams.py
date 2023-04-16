from typing import Dict, List
import cupy
from lattice import get_backend
from opt_einsum import contract

class Meson:
    def __init__(self, elemental, operator, source) -> None:
        self.elemental = elemental
        self.operator = operator
        self.dagger = source
        self.outward = 1
        self.inward = 1
        self.cache = None
        self.cache_dagger = None
        self.make_cache()

    def make_cache(self):
        from lattice.insertion.gamma import gamma
        np = get_backend()

        cache: Dict[int, np.ndarray] = {}

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
                    cache[deriv_mom_tuple] = self.elemental[:, derivative_idx, momentum_idx]
                if j == 0:
                    ret_elemental.append(elemental_coeff * cache[deriv_mom_tuple])
                else:
                    ret_elemental[-1] += elemental_coeff * cache[deriv_mom_tuple]
        self.cache = (
            np.asarray(ret_gamma),
            np.asarray(ret_elemental),
        )
        self.cache_dagger = (
            contract("ik,xlk,lj->xij", gamma(8), self.cache[0].conj(), gamma(8)),
            contract("xtba->xtab", self.cache[1].conj()),
        )

    def get(self, t):
        if isinstance(t, int):
            if self.dagger:
                return contract(
                    "xij,xab->ijab",
                    self.cache_dagger[0],
                    self.cache_dagger[1][:, t],
                )
            else:
                return contract(
                    "xij,xab->ijab",
                    self.cache[0],
                    self.cache[1][:, t],
                )
        else:
            if self.dagger:
                return contract(
                    "xij,xtab->tijab",
                    self.cache_dagger[0],
                    self.cache_dagger[1][:, t],
                )
            else:
                return contract(
                    "xij,xtab->tijab",
                    self.cache[0],
                    self.cache[1][:, t],
                )


class Propagator:
    def __init__(self, perambulator, Lt) -> None:
        self.perambulator = perambulator
        self.Lt = Lt
        self.cache = None
        self.cache_dagger = None
        self.cached_time = None

    def get(self, t_source, t_sink):
        from lattice.insertion.gamma import gamma

        if isinstance(t_source, int) and isinstance(t_sink, int):
            if self.cached_time != t_source and self.cached_time != t_sink:
                self.cache = self.perambulator[t_source]
                self.cache_dagger = contract("ik,tlkba,lj->tijab", gamma(15), self.cache.conj(), gamma(15))
                self.cached_time = t_source
            if self.cached_time == t_source:
                return self.cache[(t_sink - t_source) % self.Lt]
            else:
                return self.cache_dagger[(t_source - t_sink) % self.Lt]
        elif isinstance(t_source, int):
            if self.cached_time != t_source:
                self.cache = self.perambulator[t_source]
                self.cache_dagger = contract("ik,tlkba,lj->tijab", gamma(15), self.cache.conj(), gamma(15))
                self.cached_time = t_source
            return self.cache[(t_sink - t_source) % self.Lt]
        elif isinstance(t_sink, int):
            if self.cached_time != t_sink:
                self.cache = self.perambulator[t_sink]
                self.cache_dagger = contract("ik,tlkba,lj->tijab", gamma(15), self.cache.conj(), gamma(15))
                self.cached_time = t_sink
            return self.cache_dagger[(t_source - t_sink) % self.Lt]
        else:
            raise ValueError("At least t_source or t_sink should be int")


class PropagatorLocal:
    def __init__(self, perambulator, Lt) -> None:
        self.perambulator = perambulator
        self.Lt = Lt
        self.cache = None
        self.make_cache()

    def make_cache(self):
        self.cache = self.perambulator[0]
        for t_source in range(1, self.Lt):
            self.cache[t_source] = self.perambulator[t_source, 0]

    def get(self, t_source, t_sink):
        if isinstance(t_source, int):
            assert t_source == t_sink, "You cannot use PropagatorLocal here"
        else:
            assert (t_source == t_sink).all(), "You cannot use PropagatorLocal here"
        return self.cache[t_source]


_EIGEN_SUBS = "abcdefghijkl"
_DIRAC_SUBS = "mnopqrstuvwx"


class QuarkDiagram:
    def __init__(self, adjacency_matrix) -> None:
        self.adjacency_matrix = adjacency_matrix
        self.operands = []
        self.subscripts = []
        self.operands_data = []
        self.analyse()

    def analyse(self) -> None:
        num_vertex = len(self.adjacency_matrix)
        checked = [False] * num_vertex
        for idx in range(num_vertex):
            if checked[idx]:
                continue
            vertices = []
            lines = []
            checked[idx] = True
            queue = [idx]
            while queue != []:
                i = queue.pop(0)
                vertices.append(i)
                for j in range(num_vertex):
                    path = self.adjacency_matrix[i][j]
                    if path != 0:
                        self.adjacency_matrix[i][j] = 0
                        if isinstance(path, int):
                            if j not in vertices:
                                checked[j] = True
                                queue.append(j)
                            lines.append([path, i, j])
                        elif isinstance(path, list):
                            if j not in vertices:
                                checked[j] = True
                                queue.append(j)
                            for _path in path:
                                lines.append([_path, i, j])
                        else:
                            raise ValueError(F"Invalid value {path} in the adjacency matrix")
            # assert outward == 0 and inward == 0, "Invalid diagram with unconsistent in/outward lines"
            vertex_operands = []
            vertex_subscripts = {}
            line_operands = []
            line_subscripts = []
            node = 0
            for line in lines:
                line_operands.append(line)
                line_subscripts.append(
                    _DIRAC_SUBS[node + 1] + _EIGEN_SUBS[node + 1] + _DIRAC_SUBS[node] + _EIGEN_SUBS[node]
                )
                if line[1] not in vertex_subscripts:
                    vertex_operands.append(line[1])
                    vertex_subscripts[line[1]] = _DIRAC_SUBS[node] + _EIGEN_SUBS[node]
                else:
                    vertex_subscripts[line[1]] = _DIRAC_SUBS[node] + _EIGEN_SUBS[node] + vertex_subscripts[line[1]]
                if line[2] not in vertex_subscripts:
                    vertex_operands.append(line[2])
                    vertex_subscripts[line[2]] = _DIRAC_SUBS[node + 1] + _EIGEN_SUBS[node + 1]
                else:
                    vertex_subscripts[line[2]
                                     ] = vertex_subscripts[line[2]] + _DIRAC_SUBS[node + 1] + _EIGEN_SUBS[node + 1]
                node += 2
            for key in range(len(line_subscripts)):
                line_subscripts[key] = line_subscripts[key][0::2] + line_subscripts[key][1::2]
            for key in vertex_subscripts.keys():
                vertex_subscripts[key] = vertex_subscripts[key][0::2] + vertex_subscripts[key][1::2]
            self.operands.append([line_operands, vertex_operands])
            self.subscripts.append(",".join(line_subscripts) + "," + ",".join(vertex_subscripts.values()))


def compute_diagrams_multitime(diagrams: List[QuarkDiagram], time_list, vertex_list, line_list):
    np = get_backend()
    diagram_value = []
    for diagram in diagrams:
        diagram_value.append(1.)
        for operands, subscripts in zip(diagram.operands, diagram.subscripts):
            have_multitime = False
            subscripts = subscripts.split(",")
            idx = 0
            operands_data = []
            for item in operands[0]:
                operands_data.append(line_list[item[0]].get(time_list[item[1]], time_list[item[2]]))
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
    return np.asarray(diagram_value)


def compute_diagrams(diagrams: List[QuarkDiagram], time_list, vertex_list, line_list):
    np = get_backend()
    diagram_value = []
    for diagram in diagrams:
        diagram_value.append(1.)
        for operands, subscripts in zip(diagram.operands, diagram.subscripts):
            operands_data = []
            for item in operands[0]:
                operands_data.append(line_list[item[0]].get(time_list[item[1]], time_list[item[2]]))
            for item in operands[1]:
                operands_data.append(vertex_list[item].get(time_list[item]))
            diagram_value[-1] *= contract(subscripts, *operands_data)
    return np.asarray(diagram_value)