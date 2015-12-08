import h5py
import math
from OP_ELM import ELM
import numpy as np
import time
import sys
import json


def timeit(start_time):
    return "(%d seconds)" % (time.time() - start_time)


def train_elm(filename, batch_size, neuron_allocation=None, inputs_normalized=False):
    if not neuron_allocation:
        neuron_allocation = {100: "sigm"}
    data_file = h5py.File(filename, "r")

    timer = time.time()
    data = data_file['data']
    labels = data_file['labels']
    print("Data loaded", timeit(timer))

    # accuracy_list = []
    # for i in range(5):
    timer = time.time()
    elm = ELM(data, labels, inputs_normalized)
    for key in neuron_allocation.keys():
        elm.add_neurons(neuron_allocation[key], key)
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
    accuracy = float(skin_skin + not_skin_not_skin) / (skin_skin + not_skin_not_skin + skin_but_not_skin + not_skin_but_skin)
    print("Accuracy:", accuracy)
    print("Finished analyzing errors", timeit(timer))
    print(len(labels), "data points")
    print("True-True:", skin_skin)
    print("False-False:", not_skin_not_skin)
    print("True-False:", skin_but_not_skin)
    print("False-True:", not_skin_but_skin)
    # print(sum(accuracy_list) / len(accuracy_list))
    elm.save(data_file.filename[:data_file.filename.rindex('.')] + '.elm')


if __name__ == '__main__':
    if "-h" in sys.argv:
        print("python elm_trainer.py <filename> <batch size> [(lin|sigm|tanh)-neuron-###]")
    else:
        neuron_args = [x for x in sys.argv if "neuron" in x]
        neurons = {x.split("-")[0]: int(x.split("-")[-1]) for x in neuron_args}
        filename = sys.argv[1]
        batch_size = int(sys.argv[2])

        train_elm(filename, batch_size, neuron_allocation=neurons, inputs_normalized=False)
    # print("Inputs normalized")
    # train_elm("/Users/test/fall_2015/bigdata/project/skin_detection/data_extraction/skin_data.hdf5", 100, {"tanh": 100}, inputs_normalized=True)
    # print("Inputs not normalized")
    # train_elm("/Users/test/fall_2015/bigdata/project/skin_detection/data_extraction/skin_data.hdf5", 100, {"tanh": 100}, inputs_normalized=False)


