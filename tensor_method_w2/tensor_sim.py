import numpy as np
import sympy as sp
from gate_sim_gates import *

class TensorSim:
    def __init__ (self, bit_string:str):
        def get_indices(bit_string:str):
            indices_list = []
            for char in bit_string:
                assert char == '0' or char == '1'
                indices_list.append(int(char))
            return tuple(indices_list)

        def initialize_state(bit_string:str):
            n:int = len(bit_string)
            for char in bit_string:
                assert char == '0' or char == '1'
            tensor = np.zeros([2] * n, dtype=object)
            tensor[get_indices(bit_string)] = 1
            return tensor

        self.state = initialize_state(bit_string)

    def apply_1q_gate(self, gate, qubit):
        assert gate.shape == (2,2)
        state = self.state
        n:int = state.ndim
        assert qubit >= 0 and qubit < n
        state = np.moveaxis(state, qubit, 0)
        state = np.tensordot(gate, state, axes=([1], [0]))
        state = np.moveaxis(state, 0, qubit)
        self.state = state

    def apply_2q_gate(self, gate, qubit1:int, qubit2:int):
        assert gate.shape == (4,4)
        state = self.state
        n_qubits = state.ndim
        permute_order = [qubit1, qubit2] + [i for i in range(n_qubits) if i != qubit1 and i != qubit2]
        state = np.transpose(state, permute_order)
        original_shape = state.shape
        state = state.reshape(4, -1)
        state = np.tensordot(gate, state, axes=([1], [0]))
        state = state.reshape([2, 2] + list(original_shape[2:]))
        inverse_permute_order = np.argsort(permute_order)
        state = np.transpose(state, inverse_permute_order)
        self.state = state

    def apply_hamiltonian_from_list(self, pauli_list:list[tuple]):
        # accept this form:
        # [('IIIZZ', 1.0), ('IIZIZ', 1.0), ('ZIIIZ', 1.0), ('IIZZI', 1.0), ('IZZII', 1.0), ('ZZIII', 1.0)]

        # sum hamiltonian terms over the new state
        new_state = np.zeros(shape=self.state.shape, dtype=object)

        def get_gate(gate_str:str):
            if gate_str == 'I':
                return I
            if gate_str == 'Z':
                return Z
            raise Exception("No gate with name '{gate_str}'")

        def apply_hamiltonian_terms(term):
            gates, weights = term
            gate_indices = []
            combined_gate = np.array([[1]], dtype=object)
            for index in range(len(gates)):
                gate_str = gates[index]
                if gate_str == 'I':
                    continue
                gate_indices.append(index)
                combined_gate = np.kron(combined_gate, get_gate(gate_str))
            # indices not used in the gate
            # other_indices =  [i for i in range(len(self.state.shape)) if i not in gate_indices]
            # indices to swap with gate_indices
            used_indices = [i for i in range(len(gate_indices))]
            permuted_state = np.moveaxis(self.state, gate_indices, used_indices)
            reshaped_state = permuted_state.reshape(2 ** len(gate_indices), -1)
            updated_state = combined_gate @ reshaped_state 
            updated_state = updated_state.reshape(*(2,) * len(gate_indices), *permuted_state.shape[len(gate_indices):])
            updated_state = np.moveaxis(updated_state, used_indices, gate_indices)
            return updated_state * weights
        for term in pauli_list:
            term_state = apply_hamiltonian_terms(term)
            new_state += term_state
        self.state = new_state
    
    def get_state_vector(self):
        return np.array(sp.Matrix(self.state.reshape(-1)).evalf()).reshape(-1)
    
    def get_probabilities(self):
        return np.abs(self.get_state_vector()) ** 2

    def eval_state(self)->None:
        qubit_matrix = self.state
        qubits = qubit_matrix.ndim
        state_vector = qubit_matrix.reshape(-1)
        assert state_vector.ndim == 1
        states = [f"|{bin(i)[2:].zfill(qubits)}⟩" for i in range(2**qubits)]

        print("Final state amplitudes:")
        for idx, amplitude in enumerate(state_vector):
            if amplitude == 0:
                continue
            print(f'{states[idx]}: {amplitude}')

        print("\nFinal state probabilities:")
        for idx, amplitude in enumerate(state_vector):
            if amplitude == 0:
                continue
            print(f'{states[idx]}: {np.abs(amplitude) ** 2}')

        norm = sum(sp.Matrix(np.abs(state_vector) ** 2))
        print(f'\nNormalization check (should be 1): {norm}')

# qc = TensorSim("00000")
# qc.apply_1q_gate(H,2)
# qc.apply_2q_gate(CNOT, 2, 4)
# qc.apply_1q_gate(X,2)
# qc.eval_state()
