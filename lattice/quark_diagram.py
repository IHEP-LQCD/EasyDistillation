from typing import Dict, List

from opt_einsum import contract

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
        self.cache = None

    def load(self, key, usedNe: int = None):
        if self.key != key:
            self.key = key
            self.elemental_data = self.elemental.load(key)
            self.usedNe = usedNe
            self._make_cache()

    def _make_cache(self):
        from lattice.insertion.gamma import gamma

        backend = get_backend()

        cache: Dict[int, backend.ndarray] = {}

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
