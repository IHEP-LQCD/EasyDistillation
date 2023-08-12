from typing import List, NamedTuple
from feynman.diagrams import Diagram
from feynman import Operator, Vertex
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)

r2l_u = dict(
    style="elliptic",
    ellipse_excentricity=4.0,
    ellipse_spread=-0.1,
    arrow=True,
)
r2l_d = dict(
    style="elliptic",
    ellipse_excentricity=4.0,
    ellipse_spread=0.1,
    arrow=True,
    # arrow_param={'width': 0.05},
)
l2r_u = dict(
    style="elliptic",
    ellipse_excentricity=4.0,
    ellipse_spread=0.1,
    arrow=True,
)
l2r_d = dict(
    style="elliptic",
    ellipse_excentricity=4.0,
    ellipse_spread=-0.1,
    arrow=True,
    # arrow_param={'width': 0.05},
)
d2u_l = dict(
    style="elliptic",
    ellipse_excentricity=0.25,
    ellipse_spread=0.3,
    arrow=True,
)
d2u_r = dict(
    style="elliptic",
    ellipse_excentricity=0.25,
    ellipse_spread=-0.3,
    arrow=True,
)

u2d_l = dict(
    style="elliptic",
    ellipse_excentricity=0.25,
    ellipse_spread=-0.3,
    arrow=True,
)
u2d_r = dict(
    style="elliptic",
    ellipse_excentricity=0.25,
    ellipse_spread=0.3,
    arrow=True,
)


def draw_diagram(diagram, adjacency_matrix, operator_list, line_color_list):
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
                        raise ValueError(f"Invalid value {path} in the adjacency matrix")
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
            style["color"] = line_color_list[propagator[0]]
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


def make_operator(hadron, pos, **kwargs):
    if pos == "src":
        if hadron == "meson":
            return meson_source(**kwargs)
        elif hadron == "baryon":
            return baryon_source(**kwargs)
    elif pos == "snk":
        if hadron == "meson":
            return meson_sink(**kwargs)
        elif hadron == "baryon":
            return baryon_sink(**kwargs)
    else:
        raise ValueError(f"Invalid position: {pos}.")


ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks([])
ax.set_yticks([])

D1 = Diagram(ax)

op1 = meson_source(D1, (0.2, 0.7), 0.1, R"$\pi_1$")
op2 = meson_source(D1, (0.2, 0.3), 0.1, R"$\pi_2$")
op3 = meson_sink(D1, (0.8, 0.7), 0.1, R"$\pi_3$")
op4 = meson_sink(D1, (0.8, 0.3), 0.1, R"$\pi_4$")
# draw_diagram(D1, [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], [op1, op2, op3, op4]) # direct diagram
draw_diagram(D1, [[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 3], [1, 0, 0, 0]], [op1, op2, op3, op4], [None, "r", "b", "b"])

# op1 = baryon_source(D1, (.2, .5), 0.1, R"$N$")
# op2 = baryon_sink(D1, (.8, .5), 0.1, R"$N$")
# draw_diagram(D1, [[0, [1, 1, 1]], [0, 0]], [op1, op2])

# op1 = baryon_source(D1, (.2, .5), .1, R"$N$")
# op3 = baryon_sink(D1, (.8, .3), .1, R"$N$")
# op2 = meson_sink(D1, (.8, .7), .1, R"$\pi$")
# draw_diagram(D1, [[0, 1, [1, 1]], [0, 0, 1], [0, 0, 0]], [op1, op2, op3])

# D1.plot()
# D1.show()


def is_row_col_zero(matrix, i):
    row_all_zero = all(value == 0 for value in matrix[i])
    col_all_zero = all(matrix[row][i] == 0 for row in range(len(matrix)))

    return row_all_zero and col_all_zero


def draw_multi_diagrams(adjacency_matrix_list, vertex_attribute_list, line_color_list, save_path=None):
    if save_path is None:
        save_path = [None] * len(adjacency_matrix_list)
    for im, isave in zip(adjacency_matrix_list, save_path):
        draw_single_diagram(im, vertex_attribute_list, line_color_list, isave)


def draw_single_diagram(adjacency_matrix, vertex_attribute_list, line_color_list, save_path=None):
    visited_all = [not is_row_col_zero(adjacency_matrix, i) for i in range(len(adjacency_matrix))]
    print(visited_all)

    # do not draw unvisited vertex
    adjacency_matrix_tmp = [
        [x for x, flag0 in zip(row0, visited_all) if flag0]
        for row0 in [row for row, flag in zip(adjacency_matrix, visited_all) if flag]
    ]
    adjacency_matrix = adjacency_matrix_tmp
    print(adjacency_matrix_tmp)
    vertex_attribute_list = [i for i, flag0 in zip(vertex_attribute_list, visited_all) if flag0]
    print(vertex_attribute_list)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    # usetex
    import matplotlib as mpl

    mpl.rc("text", usetex=True)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    diagram = Diagram(ax)
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
                        raise ValueError(f"Invalid value {path} in the adjacency matrix")
        if propagators == []:
            continue

        print(propagators)

        operator_list = [None] * len(vertex_attribute_list)
        n_src_op = sum([i["pos"] == "src" for i in vertex_attribute_list])
        n_snk_op = len(vertex_attribute_list) - n_src_op
        i_src = 0
        i_snk = 0
        size = 0.1
        for iop in range(len(vertex_attribute_list)):
            pos = vertex_attribute_list[iop]["pos"]
            type = vertex_attribute_list[iop]["type"]
            name = vertex_attribute_list[iop]["name"]
            if pos == "src":
                y_tmp = (i_src - 0.5) if n_src_op // 2 == 1 else (i_src)
                xy_tmp = (0.2, 0.5 + y_tmp / 2 * 0.6)
                print(f"src: {xy_tmp}, {n_src_op}")
                i_src += 1
                if type == "meson":
                    operator_list[iop] = meson_source(diagram, xy_tmp, size, name)
                elif type == "baryon":
                    operator_list[iop] = baryon_source(diagram, xy_tmp, size, name)
                else:
                    raise ValueError(f"Invalid hadron type: {type}.")
            elif pos == "snk":
                y_tmp = (i_snk - 0.5) if n_snk_op // 2 == 1 else (i_snk)
                xy_tmp = (0.8, 0.5 + y_tmp / 2 * 0.6)
                print(f"snk: {xy_tmp}, {n_snk_op}")
                i_snk += 1
                if type == "meson":
                    operator_list[iop] = meson_sink(diagram, xy_tmp, size, name)
                elif type == "baryon":
                    operator_list[iop] = baryon_sink(diagram, xy_tmp, size, name)
                else:
                    raise ValueError(f"Invalid hadron type: {type}.")
            else:
                raise ValueError(f"Invalid position: {pos}.")

        print(propagators)
        for propagator in propagators:
            style = {}
            visited_out_idx = propagator[1]
            visited_in_indice = propagator[2]
            op_out = operator_list[visited_out_idx]
            op_in = operator_list[visited_in_indice]
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
            style["color"] = line_color_list[propagator[0]]  # add color
            diagram.line(
                op_out.vertex_out[outward_idx[visited_out_idx]],
                op_in.vertex_in[inward_idx[visited_in_indice]],
                **style,
            )
            outward_idx[visited_out_idx] += 1
            inward_idx[visited_in_indice] += 1
    diagram.plot()
    if save_path is not None:
        diagram.savefig(save_path)
    diagram.show()
