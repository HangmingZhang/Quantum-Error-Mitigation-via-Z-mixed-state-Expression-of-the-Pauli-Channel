#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pennylane as qml
from pennylane import numpy as np
import itertools
import random


# In[2]:


# Let's start by generating all the Pauli Matrices on N qubits
# We'll return a list that will hold all the Pauli Matrices
# For example: In the case of 2 qubits, there are 16 Pauli Matrices, including II, IX, ..., ZY, ZZ


class NqubitsPauliMatrices:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
    
    def get_pauli_matrices_of_n_qubits(self) -> 'PauliMatricesList':
        pauli_i_matrix_1_qubit = np.array([[1, 0], [0, 1]])
        pauli_x_matrix_1_qubit = np.array([[0, 1], [1, 0]])
        pauli_y_matrix_1_qubit = np.array([[0, -1j], [1j, 0]])
        pauli_z_matrix_1_qubit = np.array([[1, 0], [0, -1]])
        pauli_set_1_qubit = [pauli_i_matrix_1_qubit, pauli_x_matrix_1_qubit,
                             pauli_y_matrix_1_qubit, pauli_z_matrix_1_qubit]
        
        if self.n_qubits == 1:
            return pauli_set_1_qubit
        
        # generate all combinations of Pauli matrices for n qubits
        combinations = list(itertools.product(pauli_set_1_qubit, repeat=self.n_qubits))
        
        # compute the Kronecker product for each combination
        pauli_set_n_qubits = []
        for combo in combinations:
            kron_product = combo[0]
            for matrix in combo[1:]:
                kron_product = np.kron(kron_product, matrix)
            pauli_set_n_qubits.append(kron_product)
        
        return pauli_set_n_qubits


# In[3]:


# Next, we'll create several different quantum channels,
# including depolarizing channels on n qubits,
# random Pauli channels on n qubits,
# and a identity channel on n qubits.


class NqubitsChannel:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.nqubitspaulimatrices = NqubitsPauliMatrices(n_qubits)
    
    # 退极化信道
    def nqubitsdepolarizingchannel(self, p: float) -> "DepolarizingChannelKrausMatrices":
        if self.__valid_p(p):
            pauli_set_n_qubits = self.nqubitspaulimatrices.get_pauli_matrices_of_n_qubits()
            for i in range(1, len(pauli_set_n_qubits)):
                pauli_set_n_qubits[i] = pauli_set_n_qubits[i] * np.sqrt((1 - p)/(4 ** self.n_qubits - 1))
            pauli_set_n_qubits[0] = np.sqrt(p) * pauli_set_n_qubits[0]
            return pauli_set_n_qubits
        
    # 随机的泡利信道
    def nqubitsrandompaulichannel(self, p_identity=0.5) -> "PauliChannelKrausMatrices":
        if self.__valid_p_identity(p_identity):
            pauli_set_n_qubits = self.nqubitspaulimatrices.get_pauli_matrices_of_n_qubits()
            coefficient_list = self.__create_random_p_distribution(p_identity)
            for i in range(len(pauli_set_n_qubits)):
                pauli_set_n_qubits[i] = pauli_set_n_qubits[i] * coefficient_list[i]
            return pauli_set_n_qubits
    
    # 单位信道
    def nqubitsidentitychannel(self) -> "IdentityChannelKrausMatrices":
        pauli_set_n_qubits = self.nqubitspaulimatrices.get_pauli_matrices_of_n_qubits()
        for i in range(1, len(pauli_set_n_qubits)):
            pauli_set_n_qubits[i] = pauli_set_n_qubits[i] * 0
        return pauli_set_n_qubits
    
    def __valid_p_identity(self, p_identity: float):
        if p_identity < 0:
            raise ValueError("p_identity can not less than 0.")
        
        elif p_identity > 1:
            raise ValueError("p_identity can not greater than 1.")
        
        else:
            return True
        
    def __valid_p(self, p: float):
        if p < 0:
            raise ValueError("p can not less than 0.")
        
        elif p > 1:
            raise ValueError("p can not greater than 1.")
        
        else:
            return True
    
    def __create_random_p_distribution(self, p_identity):
        p_total = 1
        coefficient_list = []
        coefficient_0 = random.uniform(p_identity, p_total)
        coefficient_list.append(coefficient_0)
        p_total -= coefficient_0
        for i in range(4 ** self.n_qubits - 2):
            coefficient_i = random.uniform(0, p_total)
            coefficient_list.append(coefficient_i)
            p_total -= coefficient_i
        coefficient_list.append(p_total)
        coefficient_list = np.sqrt(coefficient_list)
        return coefficient_list


# In[4]:


class QEMZMSEPC:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.nqubitschannel = NqubitsChannel(n_qubits)

    def noise_circuit(self, circuit: 'function', *args, p: float = 1,
                       kraus_matrices_of_a_pauli_channel: list = None,
                       need_gate_noise=False, need_measurement_noise=False, **kwargs) -> 'DecoratedCircuit':
        # circuit: the original quantum circuit, how the circuit is defined is given in the testV2 file
        # args and kwargs: parameters of circuit
        # p: depolarization rate
        if self.__valid_p(p):
            if kraus_matrices_of_a_pauli_channel is None:
                kraus_matrices_of_a_pauli_channel = self.nqubitschannel.nqubitsidentitychannel()
                
            circuit(*args, **kwargs)
            
            if need_gate_noise:
                self.__add_gate_noise(p)
            if need_measurement_noise:
                self.__add_measurement_noise(kraus_matrices_of_a_pauli_channel)
            
            return qml.expval(self.__create_measurement_ops())
    
    def noise_ufolding_circuit(self, circuit: 'function', noise_factor: int, *args, p: float = 1,
                kraus_matrices_of_a_pauli_channel: list = None,
                need_gate_noise=False, need_measurement_noise=False, **kwargs) -> 'DecoratedCircuit':
        # this is the global unitary folding method
        # circuit: the original quantum circuit, how the circuit is defined is given in the testV2 file
        # args and kwargs: parameters of circuit
        # noise_factor: the noise factor can only be taken as an odd number greater than or equal to 3
        # p: depolarization rate
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


    def qemzmsepc(self, circuit: 'function', z_unmitigated: float, p: float, dev: 'qml.device', *args,
                  kraus_matrices_of_a_pauli_channel: list = None, **kwargs) -> 'MitigationMethod':
        
        if kraus_matrices_of_a_pauli_channel is None:
            kraus_matrices_of_a_pauli_channel = self.nqubitschannel.nqubitsidentitychannel()
        
        noise_factor = 3
        
        z_noise_factor_3 = qml.QNode(self.noise_ufolding_circuit, dev)(circuit, noise_factor, *args, p=p,
                        kraus_matrices_of_a_pauli_channel=kraus_matrices_of_a_pauli_channel,
                        need_gate_noise=True, need_measurement_noise=True, **kwargs)
        
        if self.__valid_stability_of_zmsepc(z_unmitigated, z_noise_factor_3):
            print("It is unstable to do error mitigation in this situation, so the unmitigated value is returned.")
            p_t = None
            return z_unmitigated, p_t
        else:
            p_u = np.sqrt(z_noise_factor_3 / z_unmitigated)
            if self.n_qubits % 2 == 0:
                p_t = p_u * 0.5 * (self.__calibration_cir1_output(dev, kraus_matrices_of_a_pauli_channel) +
                           self.__calibration_cir2_output(dev, kraus_matrices_of_a_pauli_channel))
            else:
                p_t = p_u * 0.5 * (self.__calibration_cir1_output(dev, kraus_matrices_of_a_pauli_channel) -
                           self.__calibration_cir2_output(dev, kraus_matrices_of_a_pauli_channel))
            z_mitigated = z_unmitigated / p_t
            return z_mitigated, p_t
    
    def __add_gate_noise(self, p: float):
        # p: depolarization rate
        kraus_matrices_of_a_depolarizing_channel = self.nqubitschannel.nqubitsdepolarizingchannel(p)
        qml.QubitChannel(kraus_matrices_of_a_depolarizing_channel,
                         wires=[i for i in range(self.n_qubits)])
        
    def __add_measurement_noise(self, kraus_matrices_of_a_pauli_channel: list):
        qml.QubitChannel(kraus_matrices_of_a_pauli_channel,
                        wires=[i for i in range(self.n_qubits)])
    
    def __create_measurement_ops(self):
        m = qml.PauliZ(0)
        for i in range(1, self.n_qubits):
             m = m @ qml.PauliZ(i)
        return m
    
    def __valid_p(self, p: float):
        # p: depolarization rate
        if p < 0 or p > 1:
            raise ValueError("p can not less than 0 and can not greater than 1.")
        
        else:
            return True
        
    def __valid_noise_factor(self, noise_factor: int):
        if (noise_factor - 1) % 2 != 0 or noise_factor < 3:
            raise ValueError("noise_factor can only be odd and can not less than 3 during the global folding.")
        
        else:
            return True
        
    def __valid_stability_of_zmsepc(self, z_unmitigated, z_noise_factor_3):
        if z_noise_factor_3 / z_unmitigated > 1 or z_unmitigated * z_noise_factor_3 < 0:
            return True
        else:
            return False
        
    def __calibration_cir1_output(self, dev: 'qml.device',
                                  kraus_matrices_of_a_pauli_channel: list = None) -> 'ExpectionValue':
        # get the expection value of calibration 1
        return qml.QNode(self.__calibration_cir1, dev)(kraus_matrices_of_a_pauli_channel)
    
    def __calibration_cir1(self, kraus_matrices_of_a_pauli_channel: list = None) -> 'CalibrationCircuit':
        # calibration cir 1 for estimating the parameter of measurement noise in the paper
        if kraus_matrices_of_a_pauli_channel is None:
            kraus_matrices_of_a_pauli_channel = self.nqubitschannel.nqubitsidentitychannel()
            
        self.__add_measurement_noise(kraus_matrices_of_a_pauli_channel)
        
        return qml.expval(self.__create_measurement_ops())
    
    def __calibration_cir2_output(self, dev: 'qml.device',
                                  kraus_matrices_of_a_pauli_channel: list = None) -> 'ExpectionValue':
        # get the expection value of calibration 2
        return qml.QNode(self.__calibration_cir2, dev)(kraus_matrices_of_a_pauli_channel)
    
    def __calibration_cir2(self, kraus_matrices_of_a_pauli_channel: list = None) -> 'CalibrationCircuit':
        # calibration cir 2 for estimating the parameter of measurement noise in the paper
        if kraus_matrices_of_a_pauli_channel is None:
            kraus_matrices_of_a_pauli_channel = self.nqubitschannel.nqubitsidentitychannel()
            
        for i in range(self.n_qubits):
            qml.PauliX(wires=i)
            
        self.__add_measurement_noise(kraus_matrices_of_a_pauli_channel)
        
        return qml.expval(self.__create_measurement_ops())


# In[ ]:




