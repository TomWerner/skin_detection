import h5py
import math
from OP_ELM import ELM
import numpy as np
import time
import sys
import json

def timeit(start_time):
    return "(%d seconds)" % (time.time() - start_time)


def evaluate_elm(elm_file, validation_file, batch_size):
    elm = ELM(np.zeros((0,0)), np.zeros((0,0)))
    elm.load(elm_file)

    data_file = h5py.File(validation_file, "r")

    timer = time.time()
    data = data_file['data']
    labels = data_file['labels']
    print("Data loaded", timeit(timer))

    timer = time.time()
    outer_batch_size = batch_size * 10  # How much can fit in memory at a time
    num_batches = int(math.ceil(float(data.shape[0]) / outer_batch_size))  # float division, round up

    skin_skin = 0
    skin_but_not_skin = 0
    not_skin_not_skin = 0
    not_skin_but_skin = 0

    for i in range(num_batches):
        start = i * outer_batch_size
        end = (i + 1) * outer_batch_size

        predicted_y = elm.predict(data[start: end], batch_size=batch_size, use_gpu=True)
        np.sign(predicted_y, out=predicted_y)

        current_index = 0
        
        label_batch = labels[start: end]
        for i in range(len(label_batch)):
            if label_batch[i] == 1 and predicted_y[current_index] == 1:
                skin_skin += 1
            elif label_batch[i] == -1 and predicted_y[current_index] == -1:
                not_skin_not_skin += 1
            elif label_batch[i] == -1 and predicted_y[current_index] == 1: # not_skin_but_skin
                not_skin_but_skin += 1
            elif label_batch[i] == 1 and predicted_y[current_index] == -1:
                skin_but_not_skin += 1
            else:
                print(label_batch[i], predicted_y[current_index])
            current_index += 1

    print("Finished prediction and analysis", timeit(timer))

    print("Neurons:")
    for neuron_function, num_neurons, weight_matrix, bias_vector in elm.neurons:
        print(neuron_function, num_neurons)
    print(len(labels), "data points")
    print("True-True:", skin_skin)
    print("False-False:", not_skin_not_skin)
    print("True-False:", skin_but_not_skin)
    print("False-True:", not_skin_but_skin)
    print("-" * 80)


if __name__ == '__main__':
    if "-h" in sys.argv:
        print("python elm_trainer.py <.elm file> <filename> <batch size>")
    else:
        print(sys.argv)
        evaluate_elm(elm_file=sys.argv[1], validation_file=sys.argv[2], batch_size=int(sys.argv[3]))


