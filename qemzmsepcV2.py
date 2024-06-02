#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pennylane as qml
from pennylane import numpy as np
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
        pauli_set_n_qubits = []
        pauli_set = []
        pauli_set_1_qubit = []
        pauli_set_1_qubit.append(pauli_i_matrix_1_qubit)
        pauli_set_1_qubit.append(pauli_x_matrix_1_qubit)
        pauli_set_1_qubit.append(pauli_y_matrix_1_qubit)
        pauli_set_1_qubit.append(pauli_z_matrix_1_qubit)
        for i in pauli_set_1_qubit:
            for j in pauli_set_1_qubit:
                temp = np.kron(i, j)
                pauli_set.append(temp)
        if self.n_qubits == 1:
            return pauli_set_1_qubit
        if self.n_qubits == 2:
            return pauli_set
        for _ in range(self.n_qubits - 2):
            for i in pauli_set_1_qubit:
                for j in pauli_set:
                    temp = np.kron(i, j)
                    pauli_set_n_qubits.append(temp)
            pauli_set = pauli_set_n_qubits
            pauli_set_n_qubits = []
        pauli_set_n_qubits = pauli_set.copy()
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


class QEMZMSEPC:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.nqubitschannel = NqubitsChannel(n_qubits)
    
    def noise_circuit_output(self, dev, circuit, p=1,
                       kraus_matrices_of_a_pauli_channel=None,
                       need_gate_noise=False, need_measurement_noise=False):
        
        return qml.QNode(self.__noise_circuit, dev)(circuit=circuit, p=p,
                       kraus_matrices_of_a_pauli_channel=kraus_matrices_of_a_pauli_channel,
                       need_gate_noise=need_gate_noise, need_measurement_noise=need_measurement_noise)
        
        
    def __noise_circuit(self, circuit, p,
                       kraus_matrices_of_a_pauli_channel,
                       need_gate_noise, need_measurement_noise):
        
        self.__get_noise_circuit(circuit, p,
                       kraus_matrices_of_a_pauli_channel=kraus_matrices_of_a_pauli_channel,
                       need_gate_noise=need_gate_noise, need_measurement_noise=need_measurement_noise)()
        
        m = qml.PauliZ(0)
        for i in range(1, self.n_qubits):
            m = m @ qml.PauliZ(i)
        return qml.expval(m)
    
    def __get_noise_circuit(self, circuit, p, kraus_matrices_of_a_pauli_channel,
                    need_gate_noise, need_measurement_noise):
        
        if p < 0:
            raise ValueError("p can not less than 0.")
        
        if p > 1:
            raise ValueError("p can not greater than 1.")
        
        if kraus_matrices_of_a_pauli_channel is None:
            kraus_matrices_of_a_pauli_channel = self.nqubitschannel.nqubitsidentitychannel()

        def noise_circuit():
            circuit()
            if need_gate_noise:
                kraus_matrices_of_a_depolarizing_channel = self.nqubitschannel.nqubitsdepolarizingchannel(p)
                qml.QubitChannel(K_list=kraus_matrices_of_a_depolarizing_channel,
                         wires=[i for i in range(self.n_qubits)])
                
            if need_measurement_noise:
                qml.QubitChannel(kraus_matrices_of_a_pauli_channel,
                         wires=[i for i in range(self.n_qubits)])
                
        return noise_circuit
    
    def noise_ufolding_circuit_output(self, dev, circuit, noise_factor, p=1,
                kraus_matrices_of_a_pauli_channel=None,
                need_gate_noise=False, need_measurement_noise=False):
        
        return qml.QNode(self.__noise_ufolding_circuit, dev)(circuit=circuit, noise_factor=noise_factor, p=p,
                       kraus_matrices_of_a_pauli_channel=kraus_matrices_of_a_pauli_channel,
                       need_gate_noise=need_gate_noise, need_measurement_noise=need_measurement_noise)
    
    def __noise_ufolding_circuit(self, circuit, noise_factor, p,
                kraus_matrices_of_a_pauli_channel,
                need_gate_noise, need_measurement_noise):
        
        self.__get_noise_ufolding(p=p, circuit=circuit, noise_factor=noise_factor,
                    kraus_matrices_of_a_pauli_channel=kraus_matrices_of_a_pauli_channel,
                    need_gate_noise=need_gate_noise, need_measurement_noise=need_measurement_noise)()
        
        m = qml.PauliZ(0)
        for i in range(1, self.n_qubits):
            m = m @ qml.PauliZ(i)
        return qml.expval(m)
    
    def __get_noise_ufolding(self, circuit, noise_factor, p,
                kraus_matrices_of_a_pauli_channel,
                need_gate_noise, need_measurement_noise):
        
        if (noise_factor - 1) % 2 != 0:
            raise ValueError("noise_factor can only be odd.")
            
        if noise_factor < 3:
            raise ValueError("noise_factor can not less than 3 during the global folding.")
                
        if p < 0:
            raise ValueError("p can not less than 0.")
            
        if p > 1:
            raise ValueError("p can not greater than 1.")
        
        if kraus_matrices_of_a_pauli_channel is None:
            kraus_matrices_of_a_pauli_channel = self.nqubitschannel.nqubitsidentitychannel()

        def noise_ufolding_circuit():
            
            epoch = int((noise_factor - 1) / 2)
            
            for _ in range(epoch + 1):
                circuit()
                if need_gate_noise:
                    kraus_matrices_of_a_depolarizing_channel = self.nqubitschannel.nqubitsdepolarizingchannel(p)
                    qml.QubitChannel(K_list=kraus_matrices_of_a_depolarizing_channel,
                                 wires=[i for i in range(self.n_qubits)])
    
            for _ in range(epoch):
                qml.adjoint(circuit)()
                if need_gate_noise:
                    kraus_matrices_of_a_depolarizing_channel = self.nqubitschannel.nqubitsdepolarizingchannel(p)
                    qml.QubitChannel(K_list=kraus_matrices_of_a_depolarizing_channel,
                                 wires=[i for i in range(self.n_qubits)])
    
            if need_measurement_noise:
                qml.QubitChannel(kraus_matrices_of_a_pauli_channel,
                             wires=[i for i in range(self.n_qubits)])
    
        return noise_ufolding_circuit
    
    def __calibration_cir1_output(self, dev, kraus_matrices_of_a_pauli_channel=None):
        return qml.QNode(self.__calibration_cir1, dev)(kraus_matrices_of_a_pauli_channel)
    
    def __calibration_cir1(self, kraus_matrices_of_a_pauli_channel=None):
        if kraus_matrices_of_a_pauli_channel is None:
            kraus_matrices_of_a_pauli_channel = self.nqubitschannel.nqubitsidentitychannel()
        qml.QubitChannel(kraus_matrices_of_a_pauli_channel,
                         wires=[i for i in range(self.n_qubits)])
        m = qml.PauliZ(0)
        for i in range(1, self.n_qubits):
            m = m @ qml.PauliZ(i)
        return qml.expval(m)
    
    def __calibration_cir2_output(self, dev, kraus_matrices_of_a_pauli_channel=None):
        return qml.QNode(self.__calibration_cir2, dev)(kraus_matrices_of_a_pauli_channel)
    
    def __calibration_cir2(self, kraus_matrices_of_a_pauli_channel=None):
        if kraus_matrices_of_a_pauli_channel is None:
            kraus_matrices_of_a_pauli_channel = self.nqubitschannel.nqubitsidentitychannel()
        for i in range(self.n_qubits):
            qml.PauliX(wires=i)
        qml.QubitChannel(kraus_matrices_of_a_pauli_channel,
                         wires=[i for i in range(self.n_qubits)])
        m = qml.PauliZ(0)
        for i in range(1, self.n_qubits):
            m = m @ qml.PauliZ(i)
        return qml.expval(m)

    def qemzmsepc(self, circuit, p, dev, kraus_matrices_of_a_pauli_channel=None):
        
        if kraus_matrices_of_a_pauli_channel is None:
            kraus_matrices_of_a_pauli_channel = self.nqubitschannel.nqubitsidentitychannel()

        z_unmitigated = self.noise_circuit_output(dev=dev, circuit=circuit, p=p,
                       kraus_matrices_of_a_pauli_channel=kraus_matrices_of_a_pauli_channel,
                       need_gate_noise=True, need_measurement_noise=True)
        
        z_noise_factor_3 = self.noise_ufolding_circuit_output(noise_factor=3, circuit=circuit, p=p, dev=dev,
                  kraus_matrices_of_a_pauli_channel=kraus_matrices_of_a_pauli_channel,
                  need_gate_noise=True, need_measurement_noise=True)
        
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

