import numpy as np
import sympy as sp

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