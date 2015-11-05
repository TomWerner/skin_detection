import h5py
import math
from skin_detection.op_elm.OP_ELM import ELM
import numpy as np
import time


def timeit(start_time):
    return "(%d seconds)" % (time.time() - start_time)

file = h5py.File("../data_extraction/skin_data.hdf5", "r")
batch_size = 100000  # TODO: Figure out the optimum value for hdf5 files


timer = time.time()
data = file['data']
labels = file['labels']
print("Data loaded", timeit(timer))

timer = time.time()
elm = ELM(data, labels)
elm.add_neurons(100, "tanh")
elm.add_neurons(100, "lin")
elm.add_neurons(100, "sigm")
elm.train()
print("ELM trained!", timeit(timer))


timer = time.time()
predicted_y = elm.predict(data, batch_size=batch_size)
print("ELM prediction finished", timeit(timer))
np.sign(predicted_y, out=predicted_y)

timer = time.time()
num_batches = math.ceil(labels.shape[0] / batch_size)  # float division, round up
current_index = 0
missed_points = 0
for label_batch in np.array_split(labels, num_batches):
    for i in range(len(label_batch)):
        if label_batch[i] != predicted_y[current_index]:
            missed_points += 1
        current_index += 1

print("Finished analyzing errors", timeit(timer))
print(len(labels), "data points")
print(missed_points, "incorrect points")
print("%.02f%% correct" % ((len(labels) - missed_points) / len(labels) * 100))