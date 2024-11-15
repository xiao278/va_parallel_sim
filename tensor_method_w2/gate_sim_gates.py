import numpy as np
import sympy as sp

X = np.array(
    [[0,1],
     [1,0]]
)

# TODO: properly implement imaginary
Y = np.array(
    [[0,-sp.I],
     [sp.I,0]]
)

Z = np.array(
    [[1,0],
     [0,-1]]
)

H = np.array(
    [[1,1],
     [1,-1]]
)
H = H / sp.sqrt(2)

I = np.array(
    [[1,0],
     [0,1]]
)

CNOT = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 1, 0]]
)

Q_0 = np.array(
    [[1],
     [0]]
)

Q_1 = np.array(
    [[0],
     [1]]
)