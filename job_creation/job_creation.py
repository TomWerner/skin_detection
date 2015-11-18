import sys
from os import listdir
from os.path import isfile, join
import numpy as np

def create_data_extraction_jobs(num_ensembles, image_dir, label_dir, filename_base, surrounding_pixels, python_exe, queue_name):
    # First find the number of images in this directory
    file_list = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]
    min_number = int(min(file_list, key=lambda x: int(str(x)[2: -4]))[2: -4])
    max_number = int(max(file_list, key=lambda x: int(str(x)[2: -4]))[2: -4]) + 1
    
    image_groups = np.split(np.arange(min_number, max_number), num_ensembles)
    data_ext_script_loc = "~/skin_detection/data_extraction/data_extraction.py"

    for image_group in image_groups:
        data_ext_call = " ".join([python_exe, data_ext_script_loc,
                                  str(min(image_group)), str(max(image_group)),
                                  image_dir,
                                  '-l', label_dir,
                                  '-o', "~/extracted_data/" + filename_base + "_" + str(min(image_group)) + "_" + str(max(image_group)),
                                  '-p', str(surrounding_pixels)])
        file = open("dat_extr_" + str(surrounding_pixels) + "_" + str(min(image_group)) + "_" + str(max(image_group) + ".job"))
        file.write("#!/bin/sh\n")
        file.write("# This selects which queue\n")
        file.write("#$ -q " + str(queue_name) + "\n")
        file.write("# One node. 1-16 cores on smp\n")
        file.write("#$ -pe smp 1\n")
        file.write("# Make the folder first\n")
        file.write("#$ -o /Users/twrner/outputs\n")
        file.write("#$ -e /Users/twrner/errors\n")
        file.write(data_ext_call + "\n")

        file.close()


def parse_arg(flag, sys_args, default):
    if flag in sys_args:
        return sys_args[sys_args.index(flag) + 1]
    return default

if __name__ == "__main__":
    if "-h" in sys.argv:
        print("Usage: python job_creation.py\n " +
              "--num_ensembles ##\n " +
              "--image_dir xxx --label_dir xxx\n " +
              "--v_image_dir xxx --v_label_dir xxx\n " +
              "--prompt_neurons <True/False>\n " +
              "--num_pixels ##\n " +
              "--python_exe xxx\n " +
              "--filename_base xxx\n " +
              "--queue_name xxx\n ")
    num_ensembles = int(parse_arg("--num_ensembles", sys.argv, 1))
    image_dir = parse_arg("--image_dir", sys.argv, "/Shared/bdagroup3/Original/train/")
    label_dir = parse_arg("--label_dir", sys.argv, "/Shared/bdagroup3/Skin/train/")
    validation_image_dir = parse_arg("--v_image_dir", sys.argv, "/Shared/bdagroup3/Original/val/")
    validation_label_dir = parse_arg("--v_label_dir", sys.argv, "/Shared/bdagroup3/Skin/val/")
    prompt_neurons = bool(parse_arg("--prompt_neurons", sys.argv, True))
    surrounding_pixels = int(parse_arg("--num_pixels", sys.argv, 3))
    python_exe = parse_arg("--python_exe", sys.argv, "~/anaconda/bin/python")
    filename_base = parse_arg("--filename_base", sys.argv, "skin_data")
    queue_name = parse_arg("--queue_name", sys.argv, "AL")

    create_data_extraction_jobs(num_ensembles, image_dir, label_dir, filename_base, surrounding_pixels, python_exe, queue_name)
