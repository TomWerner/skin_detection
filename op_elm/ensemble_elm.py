from OP_ELM import ELM
import numpy as np
import h5py
import os
import math
import time
import sys


def timeit(start_time):
    return "(%d seconds)" % (time.time() - start_time)


def get_elm_filenames(model_directory):
    elm_models = []
    for file in os.listdir(model_directory):
        if file.endswith(".elm"):
            elm_models.append(model_directory + file)
    return elm_models

def get_data_filenames(data_directory, data_prefix):
    data_files = []
    for file in os.listdir(data_directory):
        if file.startswith(data_prefix):
            data_files.append(data_directory + file)
    return data_files


def get_filename(file_path):
    x = str(file_path)
    return x[x.rindex('/') + 1: x.rindex('.')]


def create_jobs(data_dir, data_prefix, elm_directory, batch_size):
    elm_model_files = get_elm_filenames(elm_directory)
    data_files = get_data_filenames(data_dir, data_prefix)

    for elm_model_file in elm_model_files:
        for data_file in data_files:
            elm_model_name = get_filename(elm_model_file)
            data_name = get_filename(data_file)

            output_data_file = "partial_" + elm_model_name + "_" + data_name
            output_template = """#!/bin/sh
# This selects which queue
#$ -q UI,AL
# One node. 1-16 cores on smp
#$ -pe smp 16
# Make the folder first
#$ -o /Users/twrner/outputs
#$ -e /Users/twrner/errors
~/anaconda/bin/python ~/skin_detection/op_elm/parallel_ensemble_elm.py """ + data_file + ' ~/project_results/' + output_data_file + ".hdf5" + ' ' + elm_model_file + ' ' + str(batch_size)
            file = open("/Users/twrner/jobs/test_ens_" + output_data_file + '.job', 'w')
            file.write(output_template)
            file.close()



def parse_arg(flag, default):
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return default

if __name__ == "__main__":
    data_dir_ = parse_arg("--data_dir", "/Users/twrner/extracted_data/")
    data_prefix_ = parse_arg("--data_prefix", "tst_img__")
    elm_directory_ = parse_arg("--elm_dir", "/Users/twrner/extracted_data/")
    batch_size_ = int(parse_arg("--batch_size", "8192"))

    create_jobs(data_dir_, data_prefix_, elm_directory_, batch_size_)