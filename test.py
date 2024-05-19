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
import qemzmsepc as qem


# In[3]:


n_qubits = 4
dev = qml.device('default.mixed', wires=n_qubits)

# In the simulation, we target different rotation angles of the RX gate.
# If the ZMSEPC theory is correct,
# the values after error mitigation through the QEM-ZMSEPC method should be the same as the ideal values.

rotation_angle = [0, 0.5, 0.9, 1.3, 1.7, 2.1, 2.5]
p = 0.8 # p: Depolarization rate.
for rotation_angle_of_rx in rotation_angle:
    nqubitschannel = qem.NqubitsChannel(n_qubits)
    kraus_matrices_of_a_pauli_channel = nqubitschannel.nqubitsrandompaulichannel(p_identity=0.8)
    # The operations and paras required to simulate a Trotter step quantum circuit are as follows.
    operations = ['RX', 'RX', 'RX', 'RX', 'CNOT', 'CNOT', 'RZ', 'RZ', 'CNOT', 'CNOT', 'CNOT', 'RZ', 'CNOT']
    rotation_angle_of_rz = -0.2
    paras = [[0, rotation_angle_of_rx], [1, rotation_angle_of_rx], [2, rotation_angle_of_rx], [3, rotation_angle_of_rx], [0, 1], [2, 3], [1, rotation_angle_of_rz], [3, rotation_angle_of_rz], [0, 1], [2, 3], [1, 2], [2, rotation_angle_of_rz], [1, 2]]
    # apply QEM-ZMSEPC method
    qemzmsepc = qem.QEMZMSEPC(n_qubits)
    z_ideal = qemzmsepc.circuit_output(operations=operations, paras=paras, dev=dev)
    print(f"z_ideal is {z_ideal}")
    z_unmitigated = qemzmsepc.circuit_output(operations=operations, paras=paras, p=p, dev=dev,
                        kraus_matrices_of_a_pauli_channel=kraus_matrices_of_a_pauli_channel,
                        need_gate_noise=True, need_measurement_noise=True)
    print(f"z_unmitigated is {z_unmitigated}")
    z_mitigated, _ = qemzmsepc.qemzmsepc(operations=operations, paras=paras, p=p,
                        kraus_matrices_of_a_pauli_channel=kraus_matrices_of_a_pauli_channel, dev=dev)
    print(f"z_mitigated is {z_mitigated}")
    print("------------------")


# In[4]:


# In all cases, it can be observed that the expectation values after error mitigation
# using the QEM-ZMSEPC method are identical to the ideal values, implying the validity of the ZMSEPC theory .

