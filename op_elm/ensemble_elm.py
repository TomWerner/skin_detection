from OP_ELM import ELM
import numpy as np
import h5py
import os
import math
import time
import sys


def timeit(start_time):
    return "(%d seconds)" % (time.time() - start_time)


def load_elm_models(model_directory):
    elm_models = []
    for file in os.listdir(model_directory):
        if file.endswith(".elm"):
            elm = ELM(np.zeros((0, 0)), np.zeros((0, 0)))
            elm.load(model_directory + file)
            elm_models.append(elm_models)
    return elm_models


def predict(test_data_file, output_file, model_directory, batch_size):
    data_file = h5py.File(test_data_file, "r")
    prediction_file = h5py.File(output_file, "w")
    data = data_file['data']
    prediction = prediction_file.create_dataset("labels", (data.shape[0], 1), dtype='i')

    elm_models = load_elm_models(model_directory)

    outer_batch_size = batch_size * 16  # How much can fit in memory at a time
    num_batches = int(math.ceil(float(data.shape[0]) / outer_batch_size))  # float division, round up
    num_elms = len(elm_models)

    for i in range(num_batches):
        start = i * outer_batch_size
        end = (i + 1) * outer_batch_size
        final_predicted_y = np.zeros((end - start))

        for elm in elm_models:
            predicted_y = elm.predict(data[start: end], batch_size=batch_size)
            final_predicted_y += predicted_y / num_elms

        np.sign(final_predicted_y, out=final_predicted_y)
        prediction[start: end] = final_predicted_y

if __name__ == "__main__":
    print("Loading data from:", sys.argv[1])
    print("Saving data at:", sys.argv[2])
    print("Loading models from:", sys.argv[3])
    print("Working with a batch size of:", int(sys.argv[4]))
    predict(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
x