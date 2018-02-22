# -*- coding: utf-8 -*-

import json
import os
import keras.utils.data_utils

def load_data(name):
    origin = "http://keras-rcnn.storage.googleapis.com/{}.tar.gz".format(name)

    pathname = keras.utils.data_utils.get_file(
        fname=name,
        origin=origin,
        untar=True
    )
    
    filename = os.path.join(pathname, "training.json")

    training = get_file_data(filename, pathname)

    filename = os.path.join(pathname, "test.json")

    test = get_file_data(filename, pathname)

    return training, test

def get_file_data(filename, image_path):
    if os.path.exists(filename):
        with open(filename) as data:
            partition = json.load(data)
    else:
        partition = []
        
    for dictionary in partition:

        dictionary["image"]["pathname"] = image_path + dictionary["image"]["pathname"]
    
    return partition
