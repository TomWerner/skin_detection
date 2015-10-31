from PIL import Image
import numpy as np
import h5py
import sys


def get_pixel_class(rgb_tuple):
    if rgb_tuple == (255, 255, 255):  # no skin
        return -1
    return 1


def get_number_of_usable_pixels(filename, original_dir, surrounding_pixels=3, existing_image_handle=None):
    if existing_image_handle is None:
        existing_image_handle = Image.open(original_dir + filename + ".jpg")
    # 0 1 2 3 4 5 6 7 8 9   (size = 10)
    # x x x ^ ^ ^ ^ x x x   (4 usable pixels with three on each side
    dimensions = existing_image_handle.size
    return (dimensions[0] - 2 * surrounding_pixels) * (dimensions[1] - 2 * surrounding_pixels)


def transform_pixel(rgb_tuple):
    red_value, green_value, blue_value = rgb_tuple

    # TODO: If we want to support different extraction techniques, such as HSV, etc, do it here using the RGB values

    return [red_value, green_value, blue_value]


def get_image_data(filename, original_dir, skin_dir, surrounding_pixels=3, labeled_data=True):
    full_image = Image.open(original_dir + filename + ".jpg")
    full_pixel_data = full_image.load()

    if labeled_data:
        skin_image = Image.open(skin_dir + filename + "_s.bmp")
        skin_pixel_data = skin_image.load()

    pixels_in_image = get_number_of_usable_pixels(filename, original_dir, surrounding_pixels, full_image)
    variables_per_pixel = 3  # R, G, B

    data = np.zeros((pixels_in_image, get_num_pixel_dimensions(surrounding_pixels)))
    if labeled_data:
        labels = np.zeros((pixels_in_image, 1))
    else:
        labels = None

    # Ignore the edges of the image, and take the surrounding "surrounding_pixes", so a square
    # of size (2 * surrounding_pixels + 1) on each side
    pixel_number = 0
    for x in range(surrounding_pixels, full_image.size[0] - surrounding_pixels - 1):
        for y in range(surrounding_pixels, full_image.size[1] - surrounding_pixels - 1):
            index = 0
            for x_offset in range(-surrounding_pixels, surrounding_pixels + 1):
                for y_offset in range(-surrounding_pixels, surrounding_pixels + 1):
                    data[pixel_number][index: index + variables_per_pixel] = transform_pixel(full_pixel_data[x + x_offset, y + y_offset])
                    index += variables_per_pixel
            # only try to add label information if its labeled data
            if labeled_data and labels is not None and skin_pixel_data is not None:
                labels[pixel_number] = get_pixel_class(skin_pixel_data[x, y])
            pixel_number += 1
    return data, labels


def get_num_pixel_dimensions(surrounding_pixels=3):
    variables_per_pixel = 3  # R, G, B
    size_of_square = np.square(surrounding_pixels * 2 + 1)
    return variables_per_pixel * size_of_square


def extract_pixel_information(image_number_list,
                              original_dir,
                              skin_dir,
                              output_filename="skin_data",
                              labeled_data=True,
                              surrounding_pixels=3):
    # We need to know the size of the data before we create our HDF5 file, this way the compression can work best.
    # To do this we quickly go through all the files and use the metadata to calculate the number of usable pixels
    # and then calculate the dimensionality of the input data
    total_pixels = sum([get_number_of_usable_pixels("im%05d" % i, original_dir) for i in image_number_list])
    num_pixel_dimensions = get_num_pixel_dimensions(surrounding_pixels=surrounding_pixels)

    # Now we can create out file. If this is labeled data, we need to also create a 'labels' dataset for the labels
    f = h5py.File(output_filename + ".hdf5", "w")
    data = f.create_dataset("data", (total_pixels, num_pixel_dimensions), dtype='f')
    if labeled_data:
        labels = f.create_dataset("labels", (total_pixels, 1), dtype='i')
    print("We created a matrix of size (%d, %d)" % (total_pixels, num_pixel_dimensions))

    # Now we go through all the images and extract their information. This can take a while.
    current_index = 0
    for i in image_number_list:
        # If it isn't labeled data, image_labels will be None
        image_data, image_labels = get_image_data("im%05d" % i, original_dir, skin_dir, surrounding_pixels, labeled_data)
        data[current_index: current_index + image_data.shape[0]] = image_data
        if labeled_data and labels is not None:
            labels[current_index: current_index + image_labels.shape[0]] = image_labels
        current_index += image_data.shape[0]

        if i % 100 == 0:
            print("Done with the first %d images" % i)
    print("Done!")


def display_help_information():
    print("python data_extraction.py")
    print("  <load images 1 through #> ")
    print("  <full image directory> ")
    print("Optional: (-l labeled_image_directory) (-o output_filename) (-p # surrounding_pixels)")
    print("Example: python data_extraction.py 10 ../Original/train/ -l ../Skin/train -o train_data -p 3")
    print()


def parse_arg(flag, sys_args, default):
    if flag in sys_args:
        return sys_args[sys_args.index(flag) + 1]
    return default


if __name__ == '__main__':
    if len(sys.argv) < 3 or "-h" in sys.argv:
        display_help_information()
        exit(1)
    image_num_list = list(range(1, int(sys.argv[1]) + 1))
    original_directory = sys.argv[2]
    skin_directory = parse_arg("-l", sys.argv, default=None)
    output_file = parse_arg("-o", sys.argv, default="skin_data")
    surrounding_pixel_number = int(parse_arg("-p", sys.argv, default=3))

    # Now that we've extracted the command line args, extract the information
    extract_pixel_information(image_number_list=image_num_list,
                              original_dir=original_directory,
                              skin_dir=skin_directory,
                              output_filename=output_file,
                              labeled_data=(skin_directory is None),
                              surrounding_pixels=surrounding_pixel_number)
else:
    ORIGINAL_DIRECTORY = "/Users/test/fall_2015/bigdata/project/Original/train/"
    SKIN_DIRECTORY = "/Users/test/fall_2015/bigdata/project/Skin/train/"
    extract_pixel_information(list(range(1, 10 + 1)), ORIGINAL_DIRECTORY, SKIN_DIRECTORY)
