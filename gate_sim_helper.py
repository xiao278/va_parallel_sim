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

def cnot_matrix(ctrl_bit:int, tgt_bit:int, qubits:int) -> np.ndarray:
    assert ctrl_bit != tgt_bit
    assert ctrl_bit < qubits
    assert tgt_bit < qubits
    mat_size = 2 ** qubits
    #list of tuple
    row_swaps = []
    #interval where the control bit is flipped, last bit is 1, 2nd to last is 2, 3rd to last is 4, 4th to last is 8 etc etc
    cbit_interval = 2 ** (qubits - ctrl_bit - 1)
    tbit_interval = 2 ** (qubits - tgt_bit - 1)
    swapped_rows = set()
    for r in range(cbit_interval, mat_size, cbit_interval * 2):
        upper_r = r + cbit_interval
        for current_row in range(r, upper_r):
            if current_row in swapped_rows:
                continue
            other_row = current_row + tbit_interval
            row_swaps.append((current_row, other_row))
            swapped_rows.add(current_row)
            swapped_rows.add(other_row)
    mat = np.zeros((mat_size, mat_size), dtype=object)
    np.fill_diagonal(mat, 1)
    for swap in row_swaps:
        row1, row2 = swap
        mat[[row1,row2]] = mat[[row2,row1]]
    return mat

