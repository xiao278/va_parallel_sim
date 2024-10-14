import numpy as np
import sympy as sp
import gate_sim_helper as gsh

X = np.array(
    [[0,1],
     [1,0]]
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

q_0 = np.array(
    [[1],
     [0]]
)

q_1 = np.array(
    [[0],
     [1]]
)

# print(gsh.tensor_product(q_0, q_1))
#01 becomes 00 because of the X gate on the "2nd" bit
# print(np.matmul(gsh.resize_1qgate(X,0,2), gsh.tensor_product(q_0, q_0)))

print(np.matmul(gsh.resize_1qgate(H,1,3),gsh.init_qubits("110")))