# -*- coding: utf-8 -*-

import json
import os.path

import keras.utils.data_utils


def load_data(name):
    origin = "http://keras-rcnn.storage.googleapis.com/{}.tar.gz".format(name)

    pathname = keras.utils.data_utils.get_file(
        fname=name,
        origin=origin,
        untar=True
    )

    image_directory = os.path.join(pathname, "images")

    mask_directory = os.path.join(pathname, "masks")

    if not os.path.exists(mask_directory):
        mask_directory = None

    training_pathname = os.path.join(pathname, "training.json")

    training = get_file_data(training_pathname, image_directory, mask_directory)

    test_pathname = os.path.join(pathname, "test.json")

    test = get_file_data(test_pathname, image_directory, mask_directory)

    return training, test


def get_file_data(json_pathname, image_directory, mask_directory):
    if os.path.exists(json_pathname):
        with open(json_pathname) as data:
            dictionaries = json.load(data)
    else:
        raise ValueError

    for dictionary in dictionaries:
        image_pathname = dictionary["image"]["pathname"]

        image_pathname = os.path.join(image_directory, image_pathname)

        dictionary["image"]["pathname"] = image_pathname

        for index, instance in enumerate(dictionary["objects"]):
            mask_pathname = dictionary["objects"][index]["mask"]["pathname"]

            mask_pathname = os.path.join(mask_directory, mask_pathname)

            dictionary["objects"][index]["mask"]["pathname"] = mask_pathname

    return dictionaries
