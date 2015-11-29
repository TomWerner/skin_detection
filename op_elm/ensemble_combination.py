import sys
import os
import h5py


def combine_all_model_output(result_file, hdf5_files):
    labels_list = [h5py.File(f)['labels'] for f in hdf5_files]
    output_lengths = [data_item.shape[0] for data_item in labels_list]

    if set(len(output_lengths)) != 1:
        print("Output lengths are unequal:", output_lengths)
        return

    print("Combining the output of %d models" % len(labels_list))
    combined_output = h5py.File(result_file, 'w')
    combined = combined_output.create_dataset("labels", (output_lengths[0], 1), dtype='f')





if __name__ == "__main__":
    result_file = sys.argv[1]
    data_dir = sys.argv[2]
    data_chunk = sys.argv[3]

    hdf5_files = []
    for file in os.listdir(data_dir):
        if file.endswith(data_chunk):
            hdf5_files.append(data_dir + file)


    combine_all_model_output(result_file, hdf5_files)