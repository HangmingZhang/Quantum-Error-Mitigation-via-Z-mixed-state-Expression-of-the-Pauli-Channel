#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pennylane as qml
from pennylane import numpy as np
import qemzmsepc as qem


# In[2]:


# we introduced noise into the network and trained it without any error mitigation techniques
# to observe the behavior of the Loss function degradation during training.

n_qubits = 2
dev = qml.device('default.mixed', wires=n_qubits)

nqubitschannel = qem.NqubitsChannel(n_qubits)
nqubitspaulichannel = nqubitschannel.nqubitsrandompaulichannel(p_identity=0.85)
nqubitsdepolarizingchannel = nqubitschannel.nqubitsdepolarizingchannel(0.9)

@qml.qnode(qml.device('default.mixed', wires=n_qubits))
def train_cir_without_qem(parameters):
    qml.StronglyEntanglingLayers(weights=parameters, wires=range(2))
    qml.QubitChannel(nqubitsdepolarizingchannel, wires=[0, 1])
    qml.QubitChannel(nqubitspaulichannel, wires=[0, 1])
    return qml.expval(qml.PauliZ(0)@qml.PauliZ(1))

shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=2)
weights = np.random.random(size=shape)

def cost(x):
    return (train_cir_without_qem(x) - (-1)) ** 2

opt = qml.GradientDescentOptimizer(stepsize=0.1)
steps = 100
params = weights
loss = []

for i in range(steps):
    nqubitspaulichannel = nqubitschannel.nqubitsrandompaulichannel(p_identity=0.85)
    params = opt.step(cost, params)
    if (i + 1) % 5 == 0:
        print("Cost after step {:5d}: {: .7f}".format(i + 1, cost(params)))
    loss.append(cost(params))
print("Optimized rotation angles: {}".format(params))

# When simulating a complex scenario of time-varying noise channels,
# it can be observed that the descent of the Loss function is relatively unstable
# in the absence of error mitigation techniques.

import matplotlib.pyplot as plt
x = [i for i in range(10, 100)]
y = loss[10:]
plt.plot(x, y)
plt.show()


# In[3]:


# We set the noise channel as time-invariant and observe the results again.

n_qubits = 2
dev = qml.device('default.mixed', wires=n_qubits)

nqubitschannel = qem.NqubitsChannel(n_qubits)
nqubitspaulichannel = nqubitschannel.nqubitsrandompaulichannel(p_identity=0.85)
nqubitsdepolarizingchannel = nqubitschannel.nqubitsdepolarizingchannel(0.9)

@qml.qnode(qml.device('default.mixed', wires=n_qubits))
def train_cir_without_qem(parameters):
    qml.StronglyEntanglingLayers(weights=parameters, wires=range(2))
    qml.QubitChannel(nqubitsdepolarizingchannel, wires=[0, 1])
    qml.QubitChannel(nqubitspaulichannel, wires=[0, 1])
    return qml.expval(qml.PauliZ(0)@qml.PauliZ(1))

shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=2)
weights = np.random.random(size=shape)

def cost(x):
    return (train_cir_without_qem(x) - (-1)) ** 2

opt = qml.GradientDescentOptimizer(stepsize=0.1)
steps = 100
params = weights
loss = []

for i in range(steps):
    params = opt.step(cost, params)
    if (i + 1) % 5 == 0:
        print("Cost after step {:5d}: {: .7f}".format(i + 1, cost(params)))
    loss.append(cost(params))
print("Optimized rotation angles: {}".format(params))

import matplotlib.pyplot as plt
x = [i for i in range(10, 100)]
y = loss[10:]
plt.plot(x, y)
plt.show()


# In[4]:


# We estimate the total noise 'p_t' using the QEMZMSEPC method,
# allowing us to incorporate noise 'p_t' into the definition of the Loss function,
# thereby mitigating the impact of noise on quantum machine learning tasks.

operations = ['RX', 'RY', 'RZ', 'RX', 'RY', 'RZ', 'CNOT', 'CNOT',
              'RX', 'RY', 'RZ', 'RX', 'RY', 'RZ', 'CNOT', 'CNOT']
shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=2)
weights = np.random.random(size=shape)
w = weights.copy().reshape(-1)
paras = [[0, w[0]], [0, w[1]], [0, w[2]], [0, w[3]], [0, w[4]], [0, w[5]], [0, 1], [1, 0],
     [0, w[6]], [0, w[7]], [0, w[8]], [0, w[9]], [0, w[10]], [0, w[11]], [0, 1], [1, 0]]
qemzmsepc = qem.QEMZMSEPC(n_qubits)
_, p_t = qemzmsepc.qemzmsepc(operations=operations, paras=paras, dev=dev, p=0.9,
                         kraus_matrices_of_a_pauli_channel=nqubitspaulichannel)
print(p_t)


# In[5]:


@qml.qnode(qml.device('default.mixed', wires=n_qubits))
def train_cir_with_qem(parameters):
    qml.StronglyEntanglingLayers(weights=parameters, wires=range(2))
    qml.QubitChannel(nqubitsdepolarizingchannel, wires=[0, 1])
    qml.QubitChannel(nqubitspaulichannel, wires=[0, 1])
    return qml.expval(qml.PauliZ(0)@qml.PauliZ(1))

shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=2)
weights = np.random.random(size=shape)

# We modify the definition of the Loss function
# by dividing the output of the quantum circuit by the noise parameter p_t
# to obtain the expected output of the quantum circuit after error mitigation,
# and then proceed with the Loss calculation.

def cost(x):
    return ((train_cir_with_qem(x) / p_t) - (-1)) ** 2

opt = qml.GradientDescentOptimizer(stepsize=0.1)
steps = 100
params = weights
loss = []
qemzmsepc = qem.QEMZMSEPC(n_qubits)

for i in range(steps):
    params = opt.step(cost, params)
    w = params.copy().reshape(-1)
    paras = [[0, w[0]], [0, w[1]], [0, w[2]], [0, w[3]], [0, w[4]], [0, w[5]], [0, 1], [1, 0],
         [0, w[6]], [0, w[7]], [0, w[8]], [0, w[9]], [0, w[10]], [0, w[11]], [0, 1], [1, 0]]
    nqubitspaulichannel = nqubitschannel.nqubitsrandompaulichannel(p_identity=0.85)
    _, p_t = qemzmsepc.qemzmsepc(operations=operations, paras=paras, dev=dev, p=0.9,
                             kraus_matrices_of_a_pauli_channel=nqubitspaulichannel)
    if (i + 1) % 5 == 0:
        print("Cost after step {:5d}: {: .7f}".format(i + 1, cost(params)))
    loss.append(cost(params))
print("Optimized rotation angles: {}".format(params))
import matplotlib.pyplot as plt
x = [i for i in range(10, 100)]
y = loss[10:]
plt.plot(x, y)
plt.show()


# In[6]:


# From the results, we can observe that the Loss function exhibits a stable decreasing trend
# after error mitigation through the QEMZMSEPC scheme,
# and ultimately, the outcome is better than without error mitigation.

