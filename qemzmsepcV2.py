#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pennylane as qml
from pennylane import numpy as np
import itertools
import random


# In[2]:


class NqubitsPauliMatrices:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
    
    def get_pauli_matrices_of_n_qubits(self):
        pauli_i_matrix_1_qubit = np.array([[1, 0], [0, 1]])
        pauli_x_matrix_1_qubit = np.array([[0, 1], [1, 0]])
        pauli_y_matrix_1_qubit = np.array([[0, -1j], [1j, 0]])
        pauli_z_matrix_1_qubit = np.array([[1, 0], [0, -1]])
        pauli_set_1_qubit = [pauli_i_matrix_1_qubit, pauli_x_matrix_1_qubit,
                             pauli_y_matrix_1_qubit, pauli_z_matrix_1_qubit]
        
        if self.n_qubits == 1:
            return pauli_set_1_qubit
        
        # Generate all combinations of Pauli matrices for n qubits
        combinations = list(itertools.product(pauli_set_1_qubit, repeat=self.n_qubits))
        
        # Compute the Kronecker product for each combination
        pauli_set_n_qubits = []
        for combo in combinations:
            kron_product = combo[0]
            for matrix in combo[1:]:
                kron_product = np.kron(kron_product, matrix)
            pauli_set_n_qubits.append(kron_product)
        
        return pauli_set_n_qubits


# In[3]:


class NqubitsChannel:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.nqubitspaulimatrices = NqubitsPauliMatrices(n_qubits)
        
    def nqubitsdepolarizingchannel(self, p):
        pauli_set_n_qubits = self.nqubitspaulimatrices.get_pauli_matrices_of_n_qubits()
        kraus_matrices = pauli_set_n_qubits.copy()
        for i in range(1, len(kraus_matrices)):
            kraus_matrices[i] = kraus_matrices[i] * np.sqrt((1 - p)/(4 ** self.n_qubits - 1))
        kraus_matrices[0] = np.sqrt(p) * kraus_matrices[0]
        return kraus_matrices
    
    def nqubitsrandompaulichannel(self, p_identity=0.5):
        if p_identity < 0:
            raise ValueError("p_identity can not less than 0.")
        if p_identity > 1:
            raise ValueError("p_identity can not greater than 1.")
        pauli_set_n_qubits = self.nqubitspaulimatrices.get_pauli_matrices_of_n_qubits()
        kraus_matrices = pauli_set_n_qubits.copy()
        p_total = 1
        coefficient_0 = random.uniform(p_identity, p_total)
        kraus_matrices[0] = kraus_matrices[0] * np.sqrt(coefficient_0)
        p_total -= coefficient_0
        for i in range(1, len(kraus_matrices) - 1):
            coefficient_i = random.uniform(0, p_total)
            kraus_matrices[i] = kraus_matrices[i] * np.sqrt(coefficient_i)
            p_total -= coefficient_i
        kraus_matrices[-1] = kraus_matrices[-1] * np.sqrt(p_total)
        return kraus_matrices
    
    def nqubitsidentitychannel(self):
        pauli_set_n_qubits = self.nqubitspaulimatrices.get_pauli_matrices_of_n_qubits()
        kraus_matrices = pauli_set_n_qubits.copy()
        for i in range(1, len(kraus_matrices)):
            kraus_matrices[i] = kraus_matrices[i] * 0
        return kraus_matrices


# In[4]:


# In the V2 version of the code, we mainly updated the input method of the parameters.
class QEMZMSEPC:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.nqubitschannel = NqubitsChannel(n_qubits)
        
    def __add_gate_noise(self, p):
        kraus_matrices_of_a_depolarizing_channel = self.nqubitschannel.nqubitsdepolarizingchannel(p)
        qml.QubitChannel(kraus_matrices_of_a_depolarizing_channel,
                         wires=[i for i in range(self.n_qubits)])
        
    def __add_measurement_noise(self, kraus_matrices_of_a_pauli_channel):
        qml.QubitChannel(kraus_matrices_of_a_pauli_channel,
                        wires=[i for i in range(self.n_qubits)])
    
    def __create_measurement_ops(self):
        m = qml.PauliZ(0)
        for i in range(1, self.n_qubits):
             m = m @ qml.PauliZ(i)
        return m
    
    def __valid_p(self, p):
        if p < 0:
            raise ValueError("p can not less than 0.")
        
        elif p > 1:
            raise ValueError("p can not greater than 1.")
        
        else:
            return True
        
    def __valid_noise_factor(self, noise_factor):
        if (noise_factor - 1) % 2 != 0:
            raise ValueError("noise_factor can only be odd.")
            
        elif noise_factor < 3:
            raise ValueError("noise_factor can not less than 3 during the global folding.")
        
        else:
            return True

    def noise_circuit(self, circuit, *args, p=1,
                       kraus_matrices_of_a_pauli_channel=None,
                       need_gate_noise=False, need_measurement_noise=False, **kwargs):
        # args and kwargs: parameters of circuit
        if self.__valid_p(p):
            if kraus_matrices_of_a_pauli_channel is None:
                kraus_matrices_of_a_pauli_channel = self.nqubitschannel.nqubitsidentitychannel()
                
            circuit(*args, **kwargs)
            if need_gate_noise:
                self.__add_gate_noise(p)
            if need_measurement_noise:
                self.__add_measurement_noise(kraus_matrices_of_a_pauli_channel)
            
            return qml.expval(self.__create_measurement_ops())
    
    def noise_ufolding_circuit(self, circuit, noise_factor, *args, p=1,
                kraus_matrices_of_a_pauli_channel=None,
                need_gate_noise=False, need_measurement_noise=False, **kwargs):
        
        if self.__valid_noise_factor(noise_factor) and self.__valid_p(p):
            if kraus_matrices_of_a_pauli_channel is None:
                kraus_matrices_of_a_pauli_channel = self.nqubitschannel.nqubitsidentitychannel()
            
            epoch = int((noise_factor - 1) / 2)
                
            for _ in range(epoch + 1):
                circuit(*args, **kwargs)
                if need_gate_noise:
                    self.__add_gate_noise(p)
            for _ in range(epoch):
                qml.adjoint(circuit)(*args, **kwargs)
                if need_gate_noise:
                    self.__add_gate_noise(p)
            
            if need_measurement_noise:
                self.__add_measurement_noise(kraus_matrices_of_a_pauli_channel)
            
            return qml.expval(self.__create_measurement_ops())
    
    def __calibration_cir1_output(self, dev, kraus_matrices_of_a_pauli_channel=None):
        return qml.QNode(self.__calibration_cir1, dev)(kraus_matrices_of_a_pauli_channel)
    
    def __calibration_cir1(self, kraus_matrices_of_a_pauli_channel=None):
        if kraus_matrices_of_a_pauli_channel is None:
            kraus_matrices_of_a_pauli_channel = self.nqubitschannel.nqubitsidentitychannel()
            
        self.__add_measurement_noise(kraus_matrices_of_a_pauli_channel)
        
        return qml.expval(self.__create_measurement_ops())
    
    def __calibration_cir2_output(self, dev, kraus_matrices_of_a_pauli_channel=None):
        return qml.QNode(self.__calibration_cir2, dev)(kraus_matrices_of_a_pauli_channel)
    
    def __calibration_cir2(self, kraus_matrices_of_a_pauli_channel=None):
        if kraus_matrices_of_a_pauli_channel is None:
            kraus_matrices_of_a_pauli_channel = self.nqubitschannel.nqubitsidentitychannel()
            
        for i in range(self.n_qubits):
            qml.PauliX(wires=i)
            
        self.__add_measurement_noise(kraus_matrices_of_a_pauli_channel)
        
        return qml.expval(self.__create_measurement_ops())

    def qemzmsepc(self, circuit, p, dev, *args, kraus_matrices_of_a_pauli_channel=None, **kwargs):
        
        if kraus_matrices_of_a_pauli_channel is None:
            kraus_matrices_of_a_pauli_channel = self.nqubitschannel.nqubitsidentitychannel()

        z_unmitigated = qml.QNode(self.noise_circuit, dev)(circuit, *args, p=p,
                        kraus_matrices_of_a_pauli_channel=kraus_matrices_of_a_pauli_channel, 
                        need_gate_noise=True, need_measurement_noise=True, **kwargs)
        
        noise_factor = 3
        
        z_noise_factor_3 = qml.QNode(self.noise_ufolding_circuit, dev)(circuit, noise_factor, *args, p=p,
                        kraus_matrices_of_a_pauli_channel=kraus_matrices_of_a_pauli_channel,
                        need_gate_noise=True, need_measurement_noise=True, **kwargs)
        
        eps = 10 ** -7
        
        if np.abs(z_unmitigated) < eps:
            raise ValueError("It is unstable to do error mitigation in this situation.")
        
        if z_unmitigated * z_noise_factor_3 < 0:
            raise ValueError("It is recommended to increase shots to obtain more stable measurement results to do error mitigation.")
            
        p_u = np.sqrt(z_noise_factor_3 / z_unmitigated)
        
        if self.n_qubits % 2 == 0:
            p_t = p_u * 0.5 * (self.__calibration_cir1_output(dev, kraus_matrices_of_a_pauli_channel) +
                       self.__calibration_cir2_output(dev, kraus_matrices_of_a_pauli_channel))
        else:
            p_t = p_u * 0.5 * (self.__calibration_cir1_output(dev, kraus_matrices_of_a_pauli_channel) -
                       self.__calibration_cir2_output(dev, kraus_matrices_of_a_pauli_channel))
        z_mitigated = z_unmitigated / p_t
        return z_mitigated, p_t

