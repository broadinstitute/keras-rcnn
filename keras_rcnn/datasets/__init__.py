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
    image_path = os.path.join(pathname, "images")
    
    filename = os.path.join(pathname, "training.json")

    training = get_file_data(filename, image_path)

    filename = os.path.join(pathname, "validation.json")

    validation = get_file_data(filename, image_path)

    filename = os.path.join(pathname, "test.json")

    test = get_file_data(filename, image_path)

    return training, validation, test

def get_file_data(filename, image_path):
    if os.path.exists(filename):
        with open(filename) as data:
            partition = json.load(data)
    else:
        partition = []
        
    for dictionary in partition:
        dictionary["filename"] = os.path.join(image_path, dictionary["filename"])
    
    return partition
