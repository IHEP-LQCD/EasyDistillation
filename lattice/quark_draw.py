from typing import List, NamedTuple
from feynman.diagrams import Diagram
from feynman import Operator, Vertex
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)

r2l_u = dict(
    style='elliptic',
    ellipse_excentricity=4.0,
    ellipse_spread=-0.1,
    arrow=True,
)
r2l_d = dict(
    style='elliptic',
    ellipse_excentricity=4.0,
    ellipse_spread=0.1,
    arrow=True,
    # arrow_param={'width': 0.05},
)
l2r_u = dict(
    style='elliptic',
    ellipse_excentricity=4.0,
    ellipse_spread=0.1,
    arrow=True,
)
l2r_d = dict(
    style='elliptic',
    ellipse_excentricity=4.0,
    ellipse_spread=-0.1,
    arrow=True,
    # arrow_param={'width': 0.05},
)
d2u_l = dict(
    style='elliptic',
    ellipse_excentricity=0.25,
    ellipse_spread=0.3,
    arrow=True,
)
d2u_r = dict(
    style='elliptic',
    ellipse_excentricity=0.25,
    ellipse_spread=-0.3,
    arrow=True,
)

u2d_l = dict(
    style='elliptic',
    ellipse_excentricity=0.25,
    ellipse_spread=-0.3,
    arrow=True,
)
u2d_r = dict(
    style='elliptic',
    ellipse_excentricity=0.25,
    ellipse_spread=0.3,
    arrow=True,
)


def draw_diagram(diagram, adjacency_matrix, operator_list):
    num_vertex = len(adjacency_matrix)
    outward_idx = [0] * num_vertex
    inward_idx = [0] * num_vertex
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
                        raise ValueError(F"Invalid value {path} in the adjacency matrix")
        if propagators == []:
            continue
        print(propagators)
        for propagator in propagators:
            style = {}
            op_out = operator_list[propagator[1]]
            op_in = operator_list[propagator[2]]
            xy_out = op_out.xy
            xy_in = op_in.xy
            if xy_out[0] == xy_in[0]:
                if xy_out[1] < xy_in[1]:
                    if xy_out[0] > 0.5:
                        style = d2u_l
                    if xy_out[0] < 0.5:
                        style = d2u_r
                if xy_out[1] > xy_in[1]:
                    if xy_out[0] > 0.5:
                        style = u2d_l
                    if xy_out[0] < 0.5:
                        style = u2d_r
            elif xy_out[1] == xy_in[1]:
                if xy_out[0] < xy_in[0]:
                    style = l2r_u
                if xy_out[0] > xy_in[0]:
                    style = r2l_d
            diagram.line(
                op_out.vertex_out[outward_idx[propagator[1]]],
                op_in.vertex_in[inward_idx[propagator[2]]],
                **style,
            )
            outward_idx[propagator[1]] += 1
            inward_idx[propagator[2]] += 1


class Meson(NamedTuple):
    vertex_out: List[Vertex]
    vertex_in: List[Vertex]
    operator: Operator
    xy: tuple


class Baryon(NamedTuple):
    vertex_out: List[Vertex]
    vertex_in: List[Vertex]
    operator: Operator
    xy: tuple


def meson_source(diagram, xy, size, tag):
    vertex_out = [diagram.vertex(xy, dy=size, marker="")]
    vertex_in = [diagram.vertex(xy, dy=-size, marker="")]
    operator = diagram.operator([vertex_out[0], vertex_in[0]], c=2)
    operator.text(tag)
    return Meson(vertex_out, vertex_in, operator, xy)


def meson_sink(diagram, xy, size, tag):
    vertex_in = [diagram.vertex(xy, dy=size, marker="")]
    vertex_out = [diagram.vertex(xy, dy=-size, marker="")]
    operator = diagram.operator([vertex_in[0], vertex_out[0]], c=2)
    operator.text(tag)
    return Meson(vertex_out, vertex_in, operator, xy)


def baryon_source(diagram, xy, size, tag):
    vertex_out = []
    vertex_out.append(diagram.vertex(xy, dy=size, marker=""))
    vertex_out.append(diagram.vertex(xy, dx=size / 2, marker=""))
    vertex_out.append(diagram.vertex(xy, dy=-size, marker=""))
    operator = diagram.operator([vertex_out[0], vertex_out[2]], c=2)
    operator.text(tag)
    return Baryon(vertex_out, [], operator, xy)


def baryon_sink(diagram, xy, size, tag):
    vertex_in = []
    vertex_in.append(diagram.vertex(xy, dy=size, marker=""))
    vertex_in.append(diagram.vertex(xy, dx=-size / 2, marker=""))
    vertex_in.append(diagram.vertex(xy, dy=-size, marker=""))
    operator = diagram.operator([vertex_in[0], vertex_in[2]], c=2)
    operator.text(tag)
    return Baryon([], vertex_in, operator, xy)


ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks([])
ax.set_yticks([])

# First diagram
D1 = Diagram(ax)

op1 = meson_source(D1, (.2, .7), 0.1, R"$\pi_1$")
op2 = meson_source(D1, (.2, .3), 0.1, R"$\pi_2$")
op3 = meson_sink(D1, (.8, .7), 0.1, R"$\pi_3$")
op4 = meson_sink(D1, (.8, .3), 0.1, R"$\pi_4$")
# draw_diagram(D1, [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], [op1, op2, op3, op4])
draw_diagram(D1, [[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]], [op1, op2, op3, op4])

# op1 = baryon_source(D1, (.2, .5), 0.1, R"$N$")
# op2 = baryon_sink(D1, (.8, .5), 0.1, R"$N$")
# draw_diagram(D1, [[0, [1, 1, 1]], [0, 0]], [op1, op2])

# op1 = baryon_source(D1, (.2, .5), .1, R"$N$")
# op3 = baryon_sink(D1, (.8, .3), .1, R"$N$")
# op2 = meson_sink(D1, (.8, .7), .1, R"$\pi$")
# draw_diagram(D1, [[0, 1, [1, 1]], [0, 0, 1], [0, 0, 0]], [op1, op2, op3])

D1.plot()
