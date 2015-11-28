from OP_ELM import ELM
import numpy as np
import h5py
import math
import time
import sys


def timeit(start_time):
    return "(%d seconds)" % (time.time() - start_time)


def load_elm(model_file):
    elm = ELM(np.zeros((0, 0)), np.zeros((0, 0)))
    elm.load(model_file)
    return elm


def predict(test_data_file, output_file, elm_model_file, batch_size):
    data_file = h5py.File(test_data_file, "r")
    prediction_file = h5py.File(output_file, "w")
    data = data_file['data']
    prediction = prediction_file.create_dataset("labels", (data.shape[0], 1), dtype='i')

    outer_batch_size = batch_size * 16  # How much can fit in memory at a time
    num_batches = int(math.ceil(float(data.shape[0]) / outer_batch_size))  # float division, round up
    elm_model = load_elm(elm_model_file)

    for i in range(num_batches):
        start = i * outer_batch_size
        end = (i + 1) * outer_batch_size

        predicted_y = elm_model.predict(data[start: end], batch_size=batch_size)
        np.sign(predicted_y, out=predicted_y)
        prediction[start: end] = predicted_y
    prediction_file.close()

if __name__ == "__main__":
    print("Loading data from:", sys.argv[1])
    print("Saving data at:", sys.argv[2])
    print("Loading elm from:", sys.argv[3])
    print("Working with a batch size of:", int(sys.argv[4]))
    start_time = time.time()
    predict(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
    print("Finished", timeit(start_time))