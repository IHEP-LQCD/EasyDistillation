from typing import Dict, List
import cupy
from lattice import set_backend, get_backend
from opt_einsum import contract

set_backend(cupy)
# cupy.cuda.Device(1).use()

###############################################################################
from lattice.insertion.mom_dict import momDict_mom9
from lattice.insertion import Insertion, Operator, GammaName, DerivativeName, ProjectionName

pi_A1 = Insertion(GammaName.PI, DerivativeName.IDEN, ProjectionName.A1, momDict_mom9)
print(pi_A1[0])
op_pi = Operator("pi", [pi_A1[0](0, 0, 0)], [1])

pi2_A1 = Insertion(GammaName.PI_2, DerivativeName.IDEN, ProjectionName.A1, momDict_mom9)
print(pi2_A1[0])
op_pi2 = Operator("pi2", [pi2_A1[0](0, 0, 0)], [1])

rho_T1 = Insertion(GammaName.RHO, DerivativeName.IDEN, ProjectionName.T1, momDict_mom9)
print(rho_T1[0])
op_rho = Operator("rho", [rho_T1[0](0, 0, 0)], [1])
###############################################################################

###############################################################################
from lattice import preset

elemental = preset.ElementalNpy(
    "/dg_hpc/LQCD/shichunjiang/DATA/clqcd_nf2_clov_L16_T128_b2.0_ml-0.05862_sn2_srho0.12_gg5.65_gf5.2_usg0.780268_usf0.949104/04.meson.mom9/clqcd_nf2_clov_L16_T128_b2.0_xi5_ml-0.05862_cfg_",
    ".mom9.npy", [4, 123, 128, 70, 70], 70
)
perambulator = preset.PerambulatorNpy(
    "/dg_hpc/LQCD/shichunjiang/DATA/clqcd_nf2_clov_L16_T128_b2.0_ml-0.05862_sn2_srho0.12_gg5.65_gf5.2_usg0.780268_usf0.949104/03.perambulator.light.single.prec1e-9/clqcd_nf2_clov_L16_T128_b2.0_xi5_ml-0.05862_cfg_",
    ".peram.npy", [128, 128, 4, 4, 70, 70], 70
)

cfg = "2000"
e = elemental.load(cfg)
p = perambulator.load(cfg)


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
                    cache[deriv_mom_tuple] = self.elemental[derivative_idx, momentum_idx]
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


rho2pipi = QuarkDiagram([[0, 1, 0], [0, 0, 2], [1, 0, 0]])
pi2pi = QuarkDiagram([[0, 1], [1, 0]])
eta2eta = QuarkDiagram([[2, 0], [0, 2]])
line = Propagator(p, 128)
line_local = PropagatorLocal(p, 128)
rho_src = Meson(e, op_rho, True)
pi_snk = Meson(e, op_pi, False)
eta_src = Meson(e, op_pi2, True)
eta_snk = Meson(e, op_pi2, False)

import numpy as npo
np = get_backend()
t_snk = npo.arange(128)

twopt = np.zeros((2, 128))
for t_src in range(128):
    print(t_src)
    tmp = compute_diagrams_multitime(
        [pi2pi, eta2eta],
        [t_src, t_snk],
        [eta_src, eta_snk],
        [None, line, line_local],
    ).real
    twopt += np.roll(tmp, -t_src, 1)
twopt /= 128
twopt[1] = -twopt[0] + 2 * twopt[1]
twopt[0] = -twopt[0]
print(twopt)
print(np.arccosh((np.roll(twopt, -1, 1) + np.roll(twopt, 1, 1)) / twopt / 2))

twopt = np.zeros((1, 128))
for t_src in range(128):
    print(t_src)
    tmp = compute_diagrams_multitime(
        [rho2pipi],
        [t_src, t_snk, t_snk],
        [rho_src, pi_snk, pi_snk],
        [None, line, line_local],
    ).real
    twopt += np.roll(tmp, -t_src, 1)
twopt /= 128
print(twopt)
###############################################################################

twopt = np.asarray(
    [
        [
            217.54205116, 133.74269926, 114.74437894, 100.93964385, 90.30064801, 81.73824015, 74.62810935, 68.60007685,
            63.38864143, 58.81237176, 54.75604528, 51.12308181, 47.84190162, 44.8541799, 42.12067583, 39.6053764,
            37.27746762, 35.1143159, 33.10091243, 31.22307212, 29.46564121, 27.81713732, 26.26948446, 24.81597004,
            23.44858085, 22.16260691, 20.95199238, 19.81132076, 18.7368577, 17.72482276, 16.77130026, 15.8724752,
            15.02651814, 14.23045936, 13.48092613, 12.77608824, 12.1124633, 11.48844035, 10.9015508, 10.34966694,
            9.83119847, 9.34487438, 8.88875839, 8.46145633, 8.06183541, 7.68941377, 7.34286249, 7.0210513, 6.72270696,
            6.44660349, 6.19219249, 5.95856351, 5.7449421, 5.55077796, 5.37541561, 5.21833739, 5.0793596, 4.95755885,
            4.85282753, 4.7647274, 4.69314526, 4.63761286, 4.5982851, 4.57479485, 4.56703231, 4.57479485, 4.5982851,
            4.63761286, 4.69314526, 4.7647274, 4.85282753, 4.95755885, 5.0793596, 5.21833739, 5.37541561, 5.55077796,
            5.7449421, 5.95856351, 6.19219249, 6.44660349, 6.72270696, 7.0210513, 7.34286249, 7.68941377, 8.06183541,
            8.46145633, 8.88875839, 9.34487438, 9.83119847, 10.34966694, 10.9015508, 11.48844035, 12.1124633,
            12.77608824, 13.48092613, 14.23045936, 15.02651814, 15.8724752, 16.77130026, 17.72482276, 18.7368577,
            19.81132076, 20.95199238, 22.16260691, 23.44858085, 24.81597004, 26.26948446, 27.81713732, 29.46564121,
            31.22307212, 33.10091243, 35.1143159, 37.27746762, 39.6053764, 42.12067583, 44.8541799, 47.84190162,
            51.12308181, 54.75604528, 58.81237176, 63.38864143, 68.60007685, 74.62810935, 81.73824015, 90.30064801,
            100.93964385, 114.74437894, 133.74269926
        ],
        [
            204.00284963, 120.33047978, 101.54280355, 88.03835434, 77.76496225, 69.65324614, 63.04365912, 57.57919761,
            52.97930063, 49.06367287, 45.69881909, 42.7770046, 40.18552287, 37.88261884, 35.79792162, 33.91537318,
            32.18768863, 30.59886756, 29.12586105, 27.71957718, 26.35180185, 25.05091969, 23.78923376, 22.56169811,
            21.35880063, 20.17595919, 19.01102488, 17.86494172, 16.72036334, 15.61672485, 14.52682508, 13.47824742,
            12.47202528, 11.51096708, 10.57234534, 9.68101192, 8.82952142, 8.00638307, 7.23283607, 6.51887241,
            5.85108201, 5.26331944, 4.71298359, 4.21044582, 3.75512023, 3.37305806, 3.05118261, 2.79767669, 2.58893218,
            2.4382377, 2.32633394, 2.25775708, 2.21343965, 2.20175271, 2.21578085, 2.25596482, 2.30956988, 2.36612776,
            2.42149723, 2.47083948, 2.52297385, 2.56834667, 2.60519211, 2.63543464, 2.65899972, 2.63543464, 2.60519211,
            2.56834667, 2.52297385, 2.47083948, 2.42149723, 2.36612776, 2.30956988, 2.25596482, 2.21578085, 2.20175271,
            2.21343965, 2.25775708, 2.32633394, 2.4382377, 2.58893218, 2.79767669, 3.05118261, 3.37305806, 3.75512023,
            4.21044582, 4.71298359, 5.26331944, 5.85108201, 6.51887241, 7.23283607, 8.00638307, 8.82952142, 9.68101192,
            10.57234534, 11.51096708, 12.47202528, 13.47824742, 14.52682508, 15.61672485, 16.72036334, 17.86494172,
            19.01102488, 20.17595919, 21.35880063, 22.56169811, 23.78923376, 25.05091969, 26.35180185, 27.71957718,
            29.12586105, 30.59886756, 32.18768863, 33.91537318, 35.79792162, 37.88261884, 40.18552287, 42.7770046,
            45.69881909, 49.06367287, 52.97930063, 57.57919761, 63.04365912, 69.65324614, 77.76496225, 88.03835434,
            101.54280355, 120.33047978
        ]
    ]
)
