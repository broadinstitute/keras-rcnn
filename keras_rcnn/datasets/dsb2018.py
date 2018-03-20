# -*- coding: utf-8 -*-

import json
import os.path

import keras.utils.data_utils


def load_data():
    name = "DSB2018"

    origin = "http://keras-rcnn.storage.googleapis.com/{}.tar.gz".format(name)

    pathname = keras.utils.data_utils.get_file(
        fname=name,
        origin=origin,
        untar=True
    )

    filename = os.path.join(pathname, "training.json")

    if os.path.exists(filename):
        with open(filename) as data:
            training = json.load(data)
    else:
        training = []

    for dictionary in training:
        dictionary["image"]["pathname"] = pathname + dictionary["image"]["pathname"]

        for instance in dictionary["objects"]:
            instance["mask"]["pathname"] = pathname + instance["mask"]["pathname"]

    filename = os.path.join(pathname, "test.json")

    if os.path.exists(filename):
        with open(filename) as data:
            test = json.load(data)
    else:
        test = []

    for dictionary in test:
        dictionary["image"]["pathname"] = pathname + dictionary["image"]["pathname"]

        for instance in dictionary["objects"]:
            instance["mask"]["pathname"] = pathname + instance["mask"]["pathname"]

    return training, test
