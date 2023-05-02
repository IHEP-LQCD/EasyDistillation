graph_C_DsD = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]
graph_DsD_C = [
    [0, 0, 0, 0, 0, 1],
    [3, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
]
graph_DsD_DsD_1 = [
    [0, 0, 1, 0, 0, 0],
    [3, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]

graph_DsD_DsD_2 = [
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 2, 0, 0],
    [2, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]
# Warning: missing!!!
graph_DsD_DsD_3 = [
    [0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 2, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]
# D_D
graph_D_D_1 = [
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]
#Ds_Ds
graph_Ds_Ds_2 = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]

adj_matrix_list = [graph_C_DsD, graph_DsD_C, graph_DsD_DsD_1, graph_DsD_DsD_2, graph_DsD_DsD_3, graph_D_D_1, graph_Ds_Ds_2]

from lattice.quark_draw import draw_single_diagram, draw_multi_diagrams

draw_single_diagram(
    adjacency_matrix=graph_Ds_Ds_2,
    vertex_attribute_list=[
        dict(pos="src", type="meson", name=R"$D$"),
        dict(pos="src", type="meson", name=R"$\bar{D}^*$"),
        dict(pos="snk", type="meson", name=R"$D$"),
        dict(pos="snk", type="meson", name=R"$\bar{D}^*$"),
        dict(pos="src", type="meson", name=R"$\chi_{c1}$"),
        dict(pos="snk", type="meson", name=R"$\chi_{c1}$"),
    ],
    line_color_list=[None, "r", "b", "b"],
)

draw_multi_diagrams(
    adjacency_matrix_list=adj_matrix_list,
    vertex_attribute_list=[
        dict(pos="src", type="meson", name=R"$D$"),
        dict(pos="src", type="meson", name=R"$\bar{D}^*$"),
        dict(pos="snk", type="meson", name=R"$D$"),
        dict(pos="snk", type="meson", name=R"$\bar{D}^*$"),
        dict(pos="src", type="meson", name=R"$\chi_{c1}$"),
        dict(pos="snk", type="meson", name=R"$\chi_{c1}$"),
    ],
    line_color_list=[None, "r", "b", "b"],
    # save_path=[F"{i}.pdf" for i in range(len(adj_matrix_list))]
)
