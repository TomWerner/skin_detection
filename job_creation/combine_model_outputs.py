import h5py
import os
import numpy as np
import sys


def create_model_combination_jobs(data_dir, data_prefix, data_suffix, job_prefix, batch_size=8192):
    print(data_dir, data_prefix, batch_size)
    output_models = {}  # Maps output group to list of models in jury
    for file in os.listdir(data_dir):
        if file.startswith(data_prefix):
            x = h5py.File(data_dir + file, 'r')
            assert 'labels' in x.keys(), "Invalid partial: " + str(file)
            elm = file[file.index("skin_data_") + 10: file.index(data_suffix)]
            if "__" in str(file):
                data_group = file[file.index("__"):]
            else:
                data_group = "skin_data_validation"

            elms = []
            if data_group in output_models.keys():
                elms = output_models[data_group]
            elms.append(elm)
            output_models[data_group] = elms

    print(len(list(output_models.keys())), "data groups")
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
~/anaconda/bin/python ~/skin_detection/op_elm/ensemble_combination.py ~/project_results/""" + output_data_file + ".hdf5" + ' ' + data_dir + ' ' + key + ' ' + str(batch_size)
        file = open("/Users/twrner/jobs/" + job_prefix + key + '.job', 'w')
        file.write(output_template)
        file.close()


def parse_arg(flag, default):
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return default

if __name__ == "__main__":
    data_dir_ = parse_arg("--data_dir", "/Users/twrner/project_results/")
    data_prefix_ = parse_arg("--data_prefix", "partial_")
    batch_size_ = int(parse_arg("--batch_size", 8192))
    data_suffix_ = parse_arg("--data_suffix", "_tst_img")
    job_prefix_ = parse_arg("--job_prefix", "test_combine_")

    create_model_combination_jobs(data_dir_, data_prefix_, data_suffix_, job_prefix_, batch_size_)
