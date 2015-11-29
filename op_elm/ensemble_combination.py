import sys
import os
import h5py
import math
import numpy as np


def combine_all_model_output(result_file, hdf5_files, batch_size):
    labels_list = [h5py.File(f)['labels'] for f in hdf5_files]
    output_lengths = [data_item.shape[0] for data_item in labels_list]

    if len(set(output_lengths)) != 1:
        print("Output lengths are unequal:", output_lengths)
        return

    print("Combining the output of %d models" % len(labels_list))
    combined_output = h5py.File(result_file, 'w')
    combined = combined_output.create_dataset("labels", (output_lengths[0], 1), dtype='f')

    outer_batch_size = batch_size * 16  # How much can fit in memory at a time
    num_batches = max(1, int(math.ceil(float(output_lengths[0]) / outer_batch_size)))  # float division, round up
    num_models = len(labels_list)

    for labels in labels_list:
        for i in range(num_batches):
            start = i * outer_batch_size
            end = (i + 1) * outer_batch_size

            combined[start: end] += labels[start: end] / num_models
    for i in range(num_batches):
        start = i * outer_batch_size
        end = (i + 1) * outer_batch_size

        combined[start: end] = np.sign(combined[start: end])

    combined_output.close()
    print("Done!")


if __name__ == "__main__":
    result_file = sys.argv[1]
    data_dir = sys.argv[2]
    data_chunk = sys.argv[3]
    batch_size = int(sys.argv[4])

    hdf5_files = []
    for file in os.listdir(data_dir):
        if file.endswith(data_chunk):
            hdf5_files.append(data_dir + file)


    combine_all_model_output(result_file, hdf5_files, batch_size)