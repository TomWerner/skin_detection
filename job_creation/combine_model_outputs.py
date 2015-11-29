import h5py
import os
import numpy as np
import sys


def create_model_combination_jobs(data_dir, data_prefix, batch_size=8192):
    output_models = {}  # Maps output group to list of models in jury
    for file in os.listdir(data_dir):
        if file.startswith(data_prefix):
            x = h5py.File(data_dir + file, 'r')
            assert 'labels' in x.keys(), "Invalid partial: " + str(file)
            elm = file[file.index("skin_data_") + 10: file.index("_tst_img")]
            data_group = file[file.index("__"):]

            output_models.get(data_group, []).append(elm)

    print(len(output_models.keys()), "data groups")
    num_models = [len(output_models[key]) for key in output_models.keys()]
    if len(set(num_models)) != 1:
        print("Uncertain number of models:", output_models)
        return

    for key in output_models:
        print("Creating job for:", key)
        output_data_file = "combined_" + key
        output_template = """#!/bin/sh
# This selects which queue
#$ -q UI,AL
# One node. 1-16 cores on smp
#$ -pe smp 1
# Make the folder first
#$ -o /Users/twrner/outputs
#$ -e /Users/twrner/errors
~/anaconda/bin/python ~/skin_detection/op_elm/ensemble_combination.py ~/project_results/""" + output_data_file + ".hdf5" + ' ' + data_dir + ' ' + key + ' ' + batch_size
        file = open("/Users/twrner/jobs/test_combine_" + key + '.job', 'w')
        file.write(output_template)
        file.close()


def parse_arg(flag, default):
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return default

if __name__ == "__name__":
    data_dir = parse_arg("--data_dir", "/Users/twrner/project_results/")
    data_prefix = parse_arg("--data_prefix", "partial_")
    batch_size = int(parse_arg("--batch_size", 8192))

    create_model_combination_jobs(data_dir, data_prefix, batch_size)