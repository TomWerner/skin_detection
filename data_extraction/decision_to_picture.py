from PIL import Image
import numpy as np
import h5py
import sys
import math
import time


def get_number_of_usable_pixels(filename, original_dir, surrounding_pixels=3, existing_image_handle=None):
    if existing_image_handle is None:
        existing_image_handle = Image.open(original_dir + filename + ".jpg")
    # 0 1 2 3 4 5 6 7 8 9   (size = 10)
    # x x x ^ ^ ^ ^ x x x   (4 usable pixels with three on each side
    dimensions = existing_image_handle.size
    return (dimensions[0] - 2 * surrounding_pixels) * (dimensions[1] - 2 * surrounding_pixels)


def create_skin_mask(filename, original_dir, skin_dir, labels, surrounding_pixels=3):
    full_image = Image.open(original_dir + filename + ".jpg")
    full_pixel_data = full_image.load()

    pixels_in_image = get_number_of_usable_pixels(filename, original_dir, surrounding_pixels, full_image)

    img = Image.new('RGB', full_image.size, "white")  # create a new white image
    pixels = img.load()  # create the pixel map

    # Ignore the edges of the image, and take the surrounding "surrounding_pixes", so a square
    # of size (2 * surrounding_pixels + 1) on each side
    pixel_number = 0
    for x in range(surrounding_pixels, img.size[0] - surrounding_pixels - 1):
        for y in range(surrounding_pixels, img.size[1] - surrounding_pixels - 1):
            if np.sign(labels[pixel_number]) == 1:
                pixels[x,y] = full_pixel_data[x, y]
            pixel_number += 1

    img.save(skin_dir + filename + "_predicted.jpg")


def extract_pixel_information(image_number_list,
                              original_dir,
                              skin_dir,
                              label_filename,
                              surrounding_pixels=3):
    label_file = h5py.File(label_filename, 'r')
    labels = label_file['labels']
    start_index = 0
    for i in image_number_list:
        end_index = start_index + get_number_of_usable_pixels("im%05d" % i, original_dir)
        image_labels = labels[start_index: end_index]
        create_skin_mask("im%05d" % i, original_dir, skin_dir, image_labels, surrounding_pixels)

        start_index = end_index

        if i % 10 == 0:
            print("Done with first %d images" % i)

for start, stop in [(min(a), max(a)) for a in np.array_split(np.arange(2001, 4001), 20)]:
    extract_pixel_information(range(start, stop + 1),
                          '/Shared/bdagroup3/Original/test/',
                          '/Users/twrner/skin_test_output/',
                          '/Users/twrner/project_results/combined___' + str(start) + '_' + str(stop) + '.hdf5.hdf5')
