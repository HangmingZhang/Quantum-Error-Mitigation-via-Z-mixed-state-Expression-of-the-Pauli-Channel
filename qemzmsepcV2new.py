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
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.nqubitspaulimatrices = NqubitsPauliMatrices(n_qubits)
        
    def nqubitsdepolarizingchannel(self, p: float) -> 'KarusMatricesListofDepolarizingChannel':
        # p: depolarization rate
        pauli_set_n_qubits = self.nqubitspaulimatrices.get_pauli_matrices_of_n_qubits()
        kraus_matrices = pauli_set_n_qubits.copy()
        for i in range(1, len(kraus_matrices)):
            kraus_matrices[i] = kraus_matrices[i] * np.sqrt((1 - p)/(4 ** self.n_qubits - 1))
        kraus_matrices[0] = np.sqrt(p) * kraus_matrices[0]
        return kraus_matrices
    
    def nqubitsrandompaulichannel(self, p_identity: float = 0.5) -> 'KarusMatricesListofRandomPauliChannel':
        # p_identity: lower limit of coefficient_0
        # coefficient_0: the coefficient in front of the term in which the Pauli matrix is the identity matrix
        if self.__valid_p_identity(p_identity):
            pauli_set_n_qubits = self.nqubitspaulimatrices.get_pauli_matrices_of_n_qubits()
            kraus_matrices = pauli_set_n_qubits.copy()
            # we consider CPTP quantum channels
            p_total = 1
            # coefficient_0: the coefficient in front of the term in which the Pauli matrix is the identity matrix
            coefficient_0 = random.uniform(p_identity, p_total)
            kraus_matrices[0] = kraus_matrices[0] * np.sqrt(coefficient_0)
            p_total -= coefficient_0
            for i in range(1, len(kraus_matrices) - 1):
                coefficient_i = random.uniform(0, p_total)
                kraus_matrices[i] = kraus_matrices[i] * np.sqrt(coefficient_i)
                p_total -= coefficient_i
            kraus_matrices[-1] = kraus_matrices[-1] * np.sqrt(p_total)
            return kraus_matrices
    
    def nqubitsidentitychannel(self) -> 'KarusMatricesListofIdentityChannel':
        pauli_set_n_qubits = self.nqubitspaulimatrices.get_pauli_matrices_of_n_qubits()
        kraus_matrices = [matrix * 0 if i != 0 else matrix for i, matrix in enumerate(pauli_set_n_qubits)]
        return kraus_matrices
    
    def __valid_p_identity(self, p_identity: float):
        if p_identity < 0 or p_identity > 1:
            raise ValueError("p_identity can not less than 0 and can not greater than 1.")
        
        else:
            return True


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




