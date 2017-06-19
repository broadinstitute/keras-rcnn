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

    filename = os.path.join(pathname, "training.json")

    with open(filename) as data:
        training = json.load(data)

    filename = os.path.join(pathname, "test.json")

    with open(filename) as data:
        test = json.load(data)

    pathname = os.path.join(pathname, "images")

    for dictionary in training:
        dictionary["filename"] = os.path.join(pathname, dictionary["filename"])

    for dictionary in test:
        dictionary["filename"] = os.path.join(pathname, dictionary["filename"])

    return training, test
