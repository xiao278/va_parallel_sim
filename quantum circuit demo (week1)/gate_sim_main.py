import numpy as np
import sympy as sp
import gate_sim_helper as helper
import gate_sim_gates as gates

class QuantumCircuit:
    def __init__(self, initial_qubits:str)->None:
        self.num_qubits = len(initial_qubits)
        self.matrix = helper.init_qubits(initial_qubits)

    def apply_simple_gate(self, gate_name:str, tgt_qubit:int)->None:
        '''
        Available gates here are H, X. Note that qubit 0 is most significiant digit
        '''
        gate = None
        if gate_name == "H":
            gate = gates.H
        elif gate_name == "X":
            gate = gates.X
        else:
            print("bad gate name")
            assert False
        resized_gate = helper.resize_1qgate(gate, tgt_qubit, self.num_qubits)
        self.matrix = np.matmul(resized_gate, self.matrix)

    def apply_cnot_gate(self, control_bit:int, target_bit:int):
        cnot_matrix = helper.cnot_matrix(control_bit, target_bit, self.num_qubits)
        self.matrix = np.matmul(cnot_matrix, self.matrix)

    def get_probabilities(self)->dict:
        result = helper.eval_probability(self.matrix, self.num_qubits)
        return result