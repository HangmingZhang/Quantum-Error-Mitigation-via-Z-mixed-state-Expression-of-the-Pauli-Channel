#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Hello readers, I am Hangming Zhang, the author of "Joint Mitigation of Quantum Gate and Measurement Errors
# via the Z-mixed-state Expression of the Pauli Channel".
# I am delighted to share my code here.
# I conducted simulations using the Pennylane library, and the specific code is shown below.
# Please feel free to reach out to us with any questions at the following email addresses:
# For general inquiries: 'lit@njupt.edu.cn'
# For technical inquiries: '2552892578@qq.com' or '1222014006@njupt.edu.cn'


# In[2]:


import pennylane as qml
from pennylane import numpy as np
import random


# In[3]:


# To obtain all Pauli basis matrices on N qubits,
# we first define a class in order to accomplish this task.
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


# In[4]:


# After obtaining all Pauli bases on N qubits,
# we can define a function to obtain all Kraus matrices of a depolarizing channel on N qubits.
# Next, we define a function to randomly generate Pauli channels on N qubits,
# which will be considered as measurement noise in the subsequent analysis.
# We're going to encapsulate these functions in a class.


class NqubitsChannel:
    def __init__(self, n_qubits, pauli_set_n_qubits):
        self.n_qubits = n_qubits
        self.pauli_set_n_qubits = pauli_set_n_qubits
        
    def nqubitsdepolarizingchannel(self, p):
        kraus_matrices = self.pauli_set_n_qubits.copy()
        for i in range(1, len(kraus_matrices)):
            kraus_matrices[i] = kraus_matrices[i] * np.sqrt((1 - p)/(4 ** self.n_qubits - 1))
        kraus_matrices[0] = np.sqrt(p) * kraus_matrices[0]
        return kraus_matrices
    
    def nqubitsrandompaulichannel(self, p_identity=0.5):
        kraus_matrices = self.pauli_set_n_qubits.copy()
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
        kraus_matrices = self.pauli_set_n_qubits.copy()
        for i in range(1, len(kraus_matrices)):
            kraus_matrices[i] = kraus_matrices[i] * 0
        return kraus_matrices


# In[5]:


class QEMZMSEPC:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.nqubitspaulimatrices = NqubitsPauliMatrices
        self.nqubitschannel = NqubitsChannel
        
        
    def circuit_output(self, operations, paras, dev, p=1,
                       kraus_matrices_of_a_pauli_channel=None,
                       need_gate_noise=False, need_measurement_noise=False):

# operations: Operation can only be 'RX', 'RY', 'RZ' or 'CNOT'.

# In our protocol,
# we only accept rotation gates and CNOT gates as input parameters for acceptable gates,
# as other gates can be constructed using a combination of these gates.
# This restriction simplifies the input parameters and
# allows for a more efficient implementation of our algorithm in quantum circuits.

# Example of operations: ['RX', 'CNOT', 'RY'].

# paras: Parameters associated with each operation in operations.

# For rotation gates, these parameters include the wire and rotation angle,
# while for CNOT gates, these parameters consist of the control qubit index and the target qubit index.

# Example of paras: [[0, 0.5], [1, 0], [1, 1.6]],
# which means to apply an RX gate on qubit 0 with a rotation angle of 0.5, a CNOT gate on [1, 0]
# and an RY on qubit 1 with a rotation angle of 1.6.

# kraus_matrices_of_a_pauli_channel: This is a custom Pauli channel on N qubits, representing measurement noise.

        return qml.QNode(self.__get_circuit, dev)(operations=operations, paras=paras, p=p,
                    kraus_matrices_of_a_pauli_channel=kraus_matrices_of_a_pauli_channel,
                    need_gate_noise=need_gate_noise, need_measurement_noise=need_measurement_noise)
    
    def __get_circuit(self, operations, paras, p=1, kraus_matrices_of_a_pauli_channel=None,
                    need_gate_noise=False, need_measurement_noise=False):
        
        if p < 0:
            raise ValueError("p can not less than 0.")
        
        if p > 1:
            raise ValueError("p can not greater than 1.")
    
        for i, operation in enumerate(operations):
            if operation == 'RX':
                qml.RX(paras[i][1], wires=paras[i][0])
            elif operation == 'RY':
                qml.RY(paras[i][1], wires=paras[i][0])
            elif operation == 'RZ':
                qml.RZ(paras[i][1], wires=paras[i][0])
            elif operation == 'CNOT':
                qml.CNOT(wires=[paras[i][0], paras[i][1]])
            else:
                raise ValueError("Operation can only be 'RX', 'RY', 'RZ' or 'CNOT'.")
        if need_gate_noise:
            nqubitspaulimatrices = self.nqubitspaulimatrices(self.n_qubits)
            pauli_set_n_qubits = nqubitspaulimatrices.get_pauli_matrices_of_n_qubits()
            nqubitschannel = self.nqubitschannel(self.n_qubits, pauli_set_n_qubits)
            kraus_matrices_of_a_depolarizing_channel = nqubitschannel.nqubitsdepolarizingchannel(p)
            qml.QubitChannel(K_list=kraus_matrices_of_a_depolarizing_channel,
                         wires=[i for i in range(self.n_qubits)])
        if need_measurement_noise:
            if kraus_matrices_of_a_pauli_channel is None:
                nqubitspaulimatrices = self.nqubitspaulimatrices(self.n_qubits)
                pauli_set_n_qubits = nqubitspaulimatrices.get_pauli_matrices_of_n_qubits()
                nqubitschannel = self.nqubitschannel(self.n_qubits, pauli_set_n_qubits)
                kraus_matrices_of_a_pauli_channel = nqubitschannel.nqubitsidentitychannel()
            qml.QubitChannel(kraus_matrices_of_a_pauli_channel,
                         wires=[i for i in range(self.n_qubits)])
        
        m = qml.PauliZ(0)
        for i in range(1, self.n_qubits):
            m = m @ qml.PauliZ(i)
        
        return qml.expval(m)
    
    def ufolding_output(self, noise_factor, operations, paras, dev, p=1,
                kraus_matrices_of_a_pauli_channel=None,
                need_gate_noise=False, need_measurement_noise=False):
        
# This involves simulating the global unitary gate folding operation.

        return qml.QNode(self.__ufolding, dev)(operations=operations, paras=paras, p=p, noise_factor=noise_factor,
                    kraus_matrices_of_a_pauli_channel=kraus_matrices_of_a_pauli_channel,
                    need_gate_noise=need_gate_noise, need_measurement_noise=need_measurement_noise)
    
    def __ufolding(self, noise_factor, operations, paras, p=1, kraus_matrices_of_a_pauli_channel=None,
                 need_gate_noise=False, need_measurement_noise=False):
        
        if (noise_factor - 1) % 2 != 0:
            raise ValueError("noise_factor can only be odd.")
        
        if noise_factor < 3:
            raise ValueError("noise_factor can not less than 3 during the global folding.")
            
        if p < 0:
            raise ValueError("p can not less than 0.")
        
        if p > 1:
            raise ValueError("p can not greater than 1.")
    
        epoch = int((noise_factor - 1) / 2)
        
        for _ in range(epoch + 1):
            for i, operation in enumerate(operations):
                if operation == 'RX':
                    qml.RX(paras[i][1], wires=paras[i][0])
                elif operation == 'RY':
                    qml.RY(paras[i][1], wires=paras[i][0])
                elif operation == 'RZ':
                    qml.RZ(paras[i][1], wires=paras[i][0])
                elif operation == 'CNOT':
                    qml.CNOT(wires=[paras[i][0], paras[i][1]])
                else:
                    raise ValueError("Operation can only be 'RX', 'RY', 'RZ' or 'CNOT'.")
            
            if need_gate_noise:
                nqubitspaulimatrices = self.nqubitspaulimatrices(self.n_qubits)
                pauli_set_n_qubits = nqubitspaulimatrices.get_pauli_matrices_of_n_qubits()
                nqubitschannel = self.nqubitschannel(self.n_qubits, pauli_set_n_qubits)
                kraus_matrices_of_a_depolarizing_channel = nqubitschannel.nqubitsdepolarizingchannel(p)
                qml.QubitChannel(K_list=kraus_matrices_of_a_depolarizing_channel,
                             wires=[i for i in range(self.n_qubits)])

        for _ in range(epoch):
            for i, operation in enumerate(operations[::-1]):
                if operation == 'RX':
                    qml.RX(-paras[-i-1][1], wires=paras[-i-1][0])
                elif operation == 'RY':
                    qml.RY(-paras[-i-1][1], wires=paras[-i-1][0])
                elif operation == 'RZ':
                    qml.RZ(-paras[-i-1][1], wires=paras[-i-1][0])
                elif operation == 'CNOT':
                    qml.CNOT(wires=[paras[-i-1][0], paras[-i-1][1]])
            
            if need_gate_noise:
                nqubitspaulimatrices = self.nqubitspaulimatrices(self.n_qubits)
                pauli_set_n_qubits = nqubitspaulimatrices.get_pauli_matrices_of_n_qubits()
                nqubitschannel = self.nqubitschannel(self.n_qubits, pauli_set_n_qubits)
                kraus_matrices_of_a_depolarizing_channel = nqubitschannel.nqubitsdepolarizingchannel(p)
                qml.QubitChannel(K_list=kraus_matrices_of_a_depolarizing_channel,
                             wires=[i for i in range(self.n_qubits)])   
        
        if need_measurement_noise:
            if kraus_matrices_of_a_pauli_channel is None:
                nqubitspaulimatrices = self.nqubitspaulimatrices(self.n_qubits)
                pauli_set_n_qubits = nqubitspaulimatrices.get_pauli_matrices_of_n_qubits()
                nqubitschannel = self.nqubitschannel(self.n_qubits, pauli_set_n_qubits)
                kraus_matrices_of_a_pauli_channel = nqubitschannel.nqubitsidentitychannel()
            qml.QubitChannel(kraus_matrices_of_a_pauli_channel,
                         wires=[i for i in range(self.n_qubits)])
        
        m = qml.PauliZ(0)
        for i in range(1, self.n_qubits):
            m = m @ qml.PauliZ(i)
        
        return qml.expval(m)
    
    def __calibration_cir1_output(self, dev, kraus_matrices_of_a_pauli_channel=None):
        return qml.QNode(self.__calibration_cir1, dev)(kraus_matrices_of_a_pauli_channel)
    
    def __calibration_cir1(self, kraus_matrices_of_a_pauli_channel=None):
        if kraus_matrices_of_a_pauli_channel is None:
            nqubitspaulimatrices = self.nqubitspaulimatrices(self.n_qubits)
            pauli_set_n_qubits = nqubitspaulimatrices.get_pauli_matrices_of_n_qubits()
            nqubitschannel = self.nqubitschannel(self.n_qubits, pauli_set_n_qubits)
            kraus_matrices_of_a_pauli_channel = nqubitschannel.nqubitsidentitychannel()
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
            nqubitspaulimatrices = self.nqubitspaulimatrices(self.n_qubits)
            pauli_set_n_qubits = nqubitspaulimatrices.get_pauli_matrices_of_n_qubits()
            nqubitschannel = self.nqubitschannel(self.n_qubits, pauli_set_n_qubits)
            kraus_matrices_of_a_pauli_channel = nqubitschannel.nqubitsidentitychannel()
        for i in range(self.n_qubits):
            qml.PauliX(wires=i)
        qml.QubitChannel(kraus_matrices_of_a_pauli_channel,
                         wires=[i for i in range(self.n_qubits)])
        m = qml.PauliZ(0)
        for i in range(1, self.n_qubits):
            m = m @ qml.PauliZ(i)
        return qml.expval(m)

    def qemzmsepc(self, operations, paras, p, dev, kraus_matrices_of_a_pauli_channel=None):
        
# This is the output expectation value of the original noisy quantum circuit on the device dev.
        if kraus_matrices_of_a_pauli_channel is None:
            nqubitspaulimatrices = self.nqubitspaulimatrices(self.n_qubits)
            pauli_set_n_qubits = nqubitspaulimatrices.get_pauli_matrices_of_n_qubits()
            nqubitschannel = self.nqubitschannel(self.n_qubits, pauli_set_n_qubits)
            kraus_matrices_of_a_pauli_channel = nqubitschannel.nqubitsidentitychannel()
            
        z_unmitigated = self.circuit_output(operations=operations, paras=paras, p=p, dev=dev,
                    kraus_matrices_of_a_pauli_channel=kraus_matrices_of_a_pauli_channel,
                    need_gate_noise=True, need_measurement_noise=True)
        p_u = np.sqrt(self.ufolding_output(noise_factor=3, operations=operations, paras=paras, p=p, dev=dev,
                  kraus_matrices_of_a_pauli_channel=kraus_matrices_of_a_pauli_channel,
                  need_gate_noise=True, need_measurement_noise=True) / z_unmitigated)
        if self.n_qubits % 2 == 0:
            p_t = p_u * 0.5 * (self.__calibration_cir1_output(dev, kraus_matrices_of_a_pauli_channel) +
                       self.__calibration_cir2_output(dev, kraus_matrices_of_a_pauli_channel))
        else:
            p_t = p_u * 0.5 * (self.__calibration_cir1_output(dev, kraus_matrices_of_a_pauli_channel) -
                       self.__calibration_cir2_output(dev, kraus_matrices_of_a_pauli_channel))
        z_mitigated = z_unmitigated / p_t
        return z_mitigated, p_t

