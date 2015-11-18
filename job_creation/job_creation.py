import sys
from os import listdir
from os.path import isfile, join
import numpy as np

def create_data_extraction_jobs(num_ensembles, image_dir, label_dir, filename_base, surrounding_pixels, python_exe):
    # First find the number of images in this directory
    file_list = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]
    min_number = min(file_list, key=lambda x: int(str(x)[2: -4]))
    max_number = max(file_list, key=lambda x: int(str(x)[2: -4]))

    image_groups = np.split(np.arange(min_number, max_number), num_ensembles)
    data_ext_script_loc = "~/skin_detection/data_extraction/data_extraction.py"

    for image_group in image_groups:
        data_ext_call = " ".join([python_exe, data_ext_script_loc,
                                  min(image_group), max(image_group),
                                  image_dir,
                                  '-l', label_dir,
                                  '-o', "~/extracted_data/" + filename_base + "_" + min(image_group) + "_" + max(image_group),
                                  '-p', surrounding_pixels])
#         print("""
# #!/bin/sh
#
# # This selects which queue
# #$ -q AL
# # One node. 1-16 cores on smp
# #$ -pe smp 1
# # Make the folder first
# #$ -o /Users/twrner/outputs
# #$ -e /Users/twrner/errors
# """)
        print(data_ext_call)


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
              "--filename_base xxx\n ")
    num_ensembles = int(parse_arg("--num_ensembes", sys.argv, 1))
    image_dir = parse_arg("--image_dir", sys.argv, "/Shared/bdagroup3/Original/train/")
    label_dir = parse_arg("--label_dir", sys.argv, "/Shared/bdagroup3/Skin/train/")
    validation_image_dir = parse_arg("--v_image_dir", sys.argv, "/Shared/bdagroup3/Original/val/")
    validation_label_dir = parse_arg("--v_label_dir", sys.argv, "/Shared/bdagroup3/Skin/val/")
    prompt_neurons = bool(parse_arg("--prompt_neurons", sys.argv, True))
    surrounding_pixels = int(parse_arg("--num_pixels", sys.argv, 3))
    python_exe = parse_arg("--python_exe", sys.argv, "~/anaconda/bin/python")
    filename_base = parse_arg("--filename_base", sys.argv, "skin_data")

    create_data_extraction_jobs(num_ensembles, image_dir, label_dir, filename_base, surrounding_pixels, python_exe)