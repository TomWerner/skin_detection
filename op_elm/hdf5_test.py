import h5py
import math
from OP_ELM import ELM
import numpy as np
import time


def timeit(start_time):
    return "(%d seconds)" % (time.time() - start_time)

file = h5py.File("../data_extraction/train_1_20_hsv.hdf5", "r")
batch_size = 100000.0  # TODO: Figure out the optimum value for hdf5 files


timer = time.time()
data = file['data']
labels = file['labels']
print("Data loaded", timeit(timer))

timer = time.time()
elm = ELM(data, labels)
elm.add_neurons(500, "tanh")
elm.add_neurons(500, "lin")
elm.add_neurons(500, "sigm")
elm.train()
print("ELM trained!", timeit(timer))


timer = time.time()
predicted_y = elm.predict(data, batch_size=batch_size)
print("ELM prediction finished", timeit(timer))
np.sign(predicted_y, out=predicted_y)

timer = time.time()
num_batches = math.ceil(float(labels.shape[0]) / batch_size)  # float division, round up
current_index = 0
skin_skin = 0
skin_but_not_skin = 0
not_skin_not_skin = 0
not_skin_but_skin = 0

missed_points = 0
for label_batch in np.array_split(labels, num_batches):
    for i in range(len(label_batch)):
        if label_batch[i] == 1 and predicted_y[current_index] == 1:
            skin_skin += 1
        elif label_batch[i] == -1 and predicted_y[current_index] == -1:
            not_skin_not_skin += 1
        elif label_batch[i] == -1 and predicted_y[current_index] == 1: # not_skin_but_skin
            not_skin_but_skin += 1
        elif label_batch[i] == 1 and predicted_y[current_index] == -1:
            skin_but_not_skin += 1
        current_index += 1

print("Finished analyzing errors", timeit(timer))
print(len(labels), "data points")
print("True-True:", skin_skin)
print("False-False:", not_skin_not_skin)
print("True-False:", skin_but_not_skin)
print("False-True:", not_skin_but_skin)
