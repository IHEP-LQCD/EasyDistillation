from sympy import Matrix, sqrt, I, S
import numpy as np
import sympy as sp

Fermion_generator = {
    "c4y": Matrix(
        [
            [sqrt(2) / 2, -sqrt(2) / 2, 0, 0],
            [sqrt(2) / 2, sqrt(2) / 2, 0, 0],
            [0, 0, sqrt(2) / 2, -sqrt(2) / 2],
            [0, 0, sqrt(2) / 2, sqrt(2) / 2],
        ]
    ),
    "c4z": Matrix(
        [
            [sqrt(2) * (1 - I) / 2, 0, 0, 0],
            [0, sqrt(2) * (1 + I) / 2, 0, 0],
            [0, 0, sqrt(2) * (1 - I) / 2, 0],
            [0, 0, 0, sqrt(2) * (1 + I) / 2],
        ]
    ),
    "inviden": Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]),
}

OhD_generator = {
    "p_ref": [0, 0, 0],
    "A_1": {
        "c4y": Matrix([[1]]),
        "c4z": Matrix([[1]]),
    },
    "A_2": {
        "c4y": Matrix([[-1]]),
        "c4z": Matrix([[-1]]),
    },
    "E": {
        "c4y": Matrix([[S(1) / 2, sqrt(3) / 2], [sqrt(3) / 2, -S(1) / 2]]),
        "c4z": Matrix([[-1, 0], [0, 1]]),
    },
    "T_1": {
        "c4y": Matrix([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
        "c4z": Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    },
    "T_2": {
        "c4y": Matrix([[0, 0, -1], [0, -1, 0], [1, 0, 0]]),
        "c4z": Matrix([[0, 1, 0], [-1, 0, 0], [0, 0, -1]]),
    },
    "G_1": {
        "c4y": Matrix([[sqrt(2) / 2, -sqrt(2) / 2], [sqrt(2) / 2, sqrt(2) / 2]]),
        "c4z": Matrix([[sqrt(2) * (1 - I) / 2, 0], [0, sqrt(2) * (1 + I) / 2]]),
    },
    "G_2": {
        "c4y": Matrix([[-sqrt(2) / 2, sqrt(2) / 2], [-sqrt(2) / 2, -sqrt(2) / 2]]),
        "c4z": Matrix([[sqrt(2) * (-1 + I) / 2, 0], [0, sqrt(2) * (-1 - I) / 2]]),
    },
    "H": {
        "c4y": Matrix(
            [
                [sqrt(2) / 4, -sqrt(6) / 4, sqrt(6) / 4, -sqrt(2) / 4],
                [sqrt(6) / 4, -sqrt(2) / 4, -sqrt(2) / 4, sqrt(6) / 4],
                [sqrt(6) / 4, sqrt(2) / 4, -sqrt(2) / 4, -sqrt(6) / 4],
                [sqrt(2) / 4, sqrt(6) / 4, sqrt(6) / 4, sqrt(2) / 4],
            ]
        ),
        "c4z": Matrix(
            [
                [sqrt(2) * (-1 - I) / 2, 0, 0, 0],
                [0, sqrt(2) * (1 - I) / 2, 0, 0],
                [0, 0, sqrt(2) * (1 + I) / 2, 0],
                [0, 0, 0, sqrt(2) * (-1 + I) / 2],
            ]
        ),
    },
}

Dic4_generator = {
    "p_ref": [0, 0, 1],
    "A_1": {
        "c4z": Matrix([[1]]),
        "invc2y": Matrix([[1]]),
    },
    "A_2": {
        "c4z": Matrix([[1]]),
        "invc2y": Matrix([[-1]]),
    },
    "B_1": {
        "c4z": Matrix([[-1]]),
        "invc2y": Matrix([[1]]),
    },
    "B_2": {
        "c4z": Matrix([[-1]]),
        "invc2y": Matrix([[-1]]),
    },
    "E": {
        "c4z": Matrix([[0, -1], [1, 0]]),
        "invc2y": Matrix([[1, 0], [0, -1]]),
    },
    "G_1": {
        "c4z": Matrix([[sqrt(2) * (1 - I) / 2, 0], [0, sqrt(2) * (1 + I) / 2]]),
        "invc2y": Matrix([[0, -S((1))], [S(1), 0]]),
    },
    "G_2": {
        "c4z": Matrix([[-sqrt(2) * (1 - I) / 2, 0], [0, -sqrt(2) * (1 + I) / 2]]),
        "invc2y": Matrix([[0, -S(1)], [S(1), 0]]),
    },
}

Dic2_generator = {
    "p_ref": [0, 1, 1],
    "A_1": {
        "c2e": Matrix([[1]]),
        "invc2f": Matrix([[1]]),
    },
    "A_2": {
        "c2e": Matrix([[1]]),
        "invc2f": Matrix([[-1]]),
    },
    "B_1": {
        "c2e": Matrix([[-1]]),
        "invc2f": Matrix([[1]]),
    },
    "B_2": {
        "c2e": Matrix([[-1]]),
        "invc2f": Matrix([[-1]]),
    },
    "G": {
        "c2e": Matrix(
            [[-sqrt(2) * I / 2, -sqrt(2) / 2], [sqrt(2) / 2, sqrt(2) * I / 2]]
        ),
        "invc2f": Matrix(
            [[sqrt(2) * I / 2, -sqrt(2) / 2], [sqrt(2) / 2, -sqrt(2) * I / 2]]
        ),
    },
}

Dic3_generator = {
    "p_ref": [1, 1, 1],
    "A_1": {
        "c3delta": Matrix([[1]]),
        "invc2b": Matrix([[1]]),
    },
    "A_2": {
        "c3delta": Matrix([[1]]),
        "invc2b": Matrix([[-1]]),
    },
    "F_1": {
        "c3delta": Matrix([[-1]]),
        "invc2b": Matrix([[I]]),
    },
    "F_2": {
        "c3delta": Matrix([[1]]),
        "invc2b": Matrix([[-I]]),
    },
    "E": {
        "c3delta": Matrix([[-S(1) / 2, sqrt(3) / 2], [-sqrt(3) / 2, -S(1) / 2]]),
        "invc2b": Matrix([[-1, 0], [0, 1]]),
    },
    "G": {
        "c3delta": Matrix(
            [
                [S(1) / 2 - I / 2, -S(1) / 2 - I / 2],
                [S(1) / 2 - I / 2, S(1) / 2 + I / 2],
            ]
        ),
        "invc2b": Matrix([[0, sqrt(2) * (1 - I) / 2], [sqrt(2) * (-1 - I) / 2, 0]]),
    },
}

C4_generator1 = {
    "p_ref": [2, 1, 1],
    "A_1": {
        "c4x": Matrix([[1]]),
    },
    "A_2": {
        "c4x": Matrix([[-1]]),
    },
    "F_1": {
        "c4x": Matrix([[I]]),
    },
    "F_2": {
        "c4x": Matrix([[-I]]),
    },
}

C4_generator2 = {
    "p_ref": [0, 1, 2],
    "A_1": {
        "invc2x": Matrix([[1]]),
    },
    "A_2": {
        "invc2x": Matrix([[-1]]),
    },
    "F_1": {
        "invc2x": Matrix([[I]]),
    },
    "F_2": {
        "invc2x": Matrix([[-I]]),
    },
}

# # diagonal_generator

# Dic4_generator = {
#     "p_ref": [0, 0, 1],
#     "A_1": {
#         "c4z": Matrix([[1]]),
#         "invc2y": Matrix([[1]]),
#     },
#     "A_2": {
#         "c4z": Matrix([[1]]),
#         "invc2y": Matrix([[-1]]),
#     },
#     "B_1": {
#         "c4z": Matrix([[-1]]),
#         "invc2y": Matrix([[1]]),
#     },
#     "B_2": {
#         "c4z": Matrix([[-1]]),
#         "invc2y": Matrix([[-1]]),
#     },
#     "E": {
#         "c4z": Matrix([[-I, 0], [0, I]]),
#         "invc2y": Matrix([[0, -1], [-1, 0]]),
#     },
#     "G_1": {
#         "c4z": Matrix([[sqrt(2) * (1 - I) / 2, 0], [0, sqrt(2) * (1 + I) / 2]]),
#         "invc2y": Matrix([[0, -S((1))], [S(1), 0]]),
#     },
#     "G_2": {
#         "c4z": Matrix([[sqrt(2) * (-1 - I) / 2, 0], [0, sqrt(2) * (-1 + I) / 2]]),
#         "invc2y": Matrix([[0, S(1)], [-S(1), 0]]),
#     },
# }

# Dic2_generator = {
#     "p_ref": [0, 1, 1],
#     "A_1": {
#         "c2e": Matrix([[1]]),
#         "invc2f": Matrix([[1]]),
#     },
#     "A_2": {
#         "c2e": Matrix([[1]]),
#         "invc2f": Matrix([[-1]]),
#     },
#     "B_1": {
#         "c2e": Matrix([[-1]]),
#         "invc2f": Matrix([[1]]),
#     },
#     "B_2": {
#         "c2e": Matrix([[-1]]),
#         "invc2f": Matrix([[-1]]),
#     },
#     "G": {
#         "c2e": Matrix([[-I, 0], [0, I]]),
#         "invc2f": Matrix([[0, I], [I, 0]]),
#     },
# }

# Dic3_generator = {
#     "p_ref": [1, 1, 1],
#     "A_1": {
#         "c3delta": Matrix([[1]]),
#         "invc2b": Matrix([[1]]),
#     },
#     "A_2": {
#         "c3delta": Matrix([[1]]),
#         "invc2b": Matrix([[-1]]),
#     },
#     "F_1": {
#         "c3delta": Matrix([[-1]]),
#         "invc2b": Matrix([[I]]),
#     },
#     "F_2": {
#         "c3delta": Matrix([[1]]),
#         "invc2b": Matrix([[-I]]),
#     },
#     "E": {
#         "c3delta": Matrix(
#             [[-1 / 2 - sqrt(3) * I / 2, 0], [0, -1 / 2 + sqrt(3) * I / 2]]
#         ),
#         "invc2b": Matrix([[0, 1], [1, 0]]),
#     },
#     "G": {
#         "c3delta": Matrix([[1 / 2 - sqrt(3) * I / 2, 0], [0, 1 / 2 + sqrt(3) * I / 2]]),
#         "invc2b": Matrix([[0, I], [I, 0]]),
#     },
# }

refRotateDict = {
    "0,0,0": "iden",
    "0,0,1": "iden",
    "0,0,-1": "c2x",
    "1,0,0": "c4y",
    "-1,0,0": "c4y^-1",
    "0,-1,0": "c4x",
    "0,1,0": "c4x^-1",
    "0,1,1": "iden",
    "0,-1,-1": "c2x",
    "0,1,-1": "c4x^-1",
    "0,-1,1": "c4x",
    "1,0,1": "c4z^-1",
    "-1,0,-1": "c2b",
    "1,0,-1": "c2a",
    "-1,0,1": "c4z",
    "1,1,0": "c4y",
    "-1,-1,0": "c2d",
    "1,-1,0": "c2c",
    "-1,1,0": "c4y^-1",
    "1,1,1": "iden",
    "1,-1,1": "c4x",
    "1,1,-1": "c4y",
    "1,-1,-1": "c2x",
    "-1,1,1": "c4z",
    "-1,1,-1": "c2y",
    "-1,-1,1": "c2z",
    "-1,-1,-1": "c2d",
    "2,1,1": "iden",
    "2,-1,1": "c4x",
    "2,-1,-1": "c2x",
    "2,1,-1": "c4x^-1",
    "1,1,-2": "c4y",
    "-2,1,-1": "c2y",
    "-1,1,2": "c4y^-1",
    "-1,2,1": "c4z",
    "-2,-1,1": "c2z",
    "1,-2,1": "c4z^-1",
    "1,2,1": "c3delta",
    "1,1,2": "c3delta^-1",
    "-1,-2,1": "c3gamma",
    "-1,1,-2": "c3gamma^-1",
    "1,-2,-1": "c3beta",
    "-1,-1,2": "c3beta^-1",
    "-1,2,-1": "c3alpha",
    "1,-1,-2": "c3alpha^1",
    "-2,1,1": "c2e",
    "-2,-1,-1": "c2f",
    "1,-1,2": "c2c",
    "-1,-1,-2": "c2d",
    "1,2,-1": "c2a",
    "-1,-2,-1": "c2b",
    "0,1,2": "iden",
    "0,-2,1": "c4x",
    "0,-1,-2": "c2x",
    "0,2,-1": "c4x^-1",
    "2,1,0": "c4y",
    "0,1,-2": "c2y",
    "-2,1,0": "c4y^-1",
    "-1,0,2": "c4z",
    "0,-1,2": "c2z",
    "1,0,2": "c4z^-1",
    "2,0,1": "c3delta",
    "1,2,0": "c3delta^-1",
    "-2,0,1": "c3gamma",
    "-1,2,0": "c3gamma^-1",
    "2,0,-1": "c3beta",
    "-1,-2,0": "c3beta^-1",
    "-2,0,-1": "c3alpha",
    "1,-2,0": "c3alpha^1",
    "0,2,1": "c2e",
    "0,-2,-1": "c2f",
    "2,-1,0": "c2c",
    "-2,-1,0": "c2d",
    "1,0,-2": "c2a",
    "-1,0,-2": "c2b",
}


falvor2irrep = {
    1: [np.array([[1, 0], [0, 1]]) / sp.sqrt(2)],
    3: [
        np.array([[0, 1], [0, 0]]),
        np.array([[0, 0], [1, 0]]),
        np.array([[1, 0], [0, -1]]) / sp.sqrt(2),
    ],
}

falvor3irrep = {
    1: [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) / sp.sqrt(3)],
    8: [
        np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
        np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]]),
        np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
        np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]),
        np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]]),
        np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]) / sp.sqrt(2),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / sp.sqrt(6),
    ],
}
