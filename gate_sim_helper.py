import numpy as np
import sympy as sp
import gate_sim_gates as gates

# Tensor product
def tensor_product(A:np.ndarray, B:np.ndarray) -> np.ndarray:
    A_m, A_n = A.shape
    B_m, B_n = B.shape
    result_arr = np.ndarray(shape=(A_m * B_m, A_n * B_n), dtype=object)
    for row in range(A_m):
        for col in range(A_n):
            res_start_row = row * B_m
            res_start_col = col * B_n
            result_arr[res_start_row: res_start_row + B_m, res_start_col: res_start_col + B_n] = A[row,col] * B
    return result_arr

def resize_1qgate(gate:np.ndarray, pos:int, qubits:int) -> np.ndarray:
    '''
    Returns the expanded version of one-qubit gates. Position 0 is most significiant digit
    '''
    assert pos < qubits
    result = gate if pos == 0 else gates.I
    for i in range(1, qubits):
        new_gate = gate if pos == i else gates.I
        result = tensor_product(result, new_gate)
    return result
    
def init_qubits(bits:str)->np.ndarray:
    for char in bits:
        assert char == '0' or char == '1'
    result = gates.Q_0 if bits[0] == '0' else gates.Q_1
    for i in range(1, len(bits)):
        next_bit = gates.Q_0 if bits[i] == '0' else gates.Q_1
        result = tensor_product(result, next_bit)
    return result

def eval_probability(qubit_matrix:np.ndarray, qubits:int)->np.ndarray:
    assert qubit_matrix.shape[1] == 1
    possibilities:int = qubit_matrix.shape[0]
    squared = np.square(qubit_matrix)
    # print(squared)
    probabilities = {}
    for i in range(possibilities):
        binary:str = ("{0:0%db}"%(qubits)).format(i)
        if squared[i,0] > 0:
            probabilities[binary] = squared[i,0]
    return probabilities