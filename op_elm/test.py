import numpy as np
from skin_detection.op_elm.OP_ELM import ELM as MY_ELM
from hpelm.elm import ELM as HIS_ELM
from scipy.linalg import solve

N = 1000 # number of data points
t = np.linspace(0, 4*np.pi, N)
data = t #np.array(3.0 * np.sin(t + 0.001) + 0.5 + np.random.randn(N))  # create artificial data with noise
outputs = np.array(3.0 * np.sin(t + 0.001) + 0.5) + np.random.randn(N)

data = data.reshape((len(data), 1))
outputs = outputs.reshape((len(outputs), 1))



np.random.seed(1)
elm = MY_ELM(data, outputs)
elm.add_neurons(30, "lin")
elm.add_neurons(30, "sigm")
elm.add_neurons(30, "tanh")
elm.train()
Y = elm.predict(data)
error = np.square(Y - outputs)

print("Mine:", float(sum(error)))

np.random.seed(1)
elm = HIS_ELM(data.shape[1], outputs.shape[1])
elm.add_neurons(30, "lin")
elm.add_neurons(30, "sigm")
elm.add_neurons(30, "tanh")
elm.train(data, outputs)
Y = elm.predict(data)
print("HIS: ", float(sum(np.square(Y - outputs))))

