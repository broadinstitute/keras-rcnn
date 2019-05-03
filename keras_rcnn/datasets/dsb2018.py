# -*- coding: utf-8 -*-

import json
import os.path

import keras.utils.data_utils


def load_data():
    origin = "http://keras-rcnn.storage.googleapis.com/{}.tar.gz".format("DSB2018")

    pathname = keras.utils.data_utils.get_file(
        fname="DSB2018",
        origin=origin,
        untar=True
    )

    filename = os.path.join(pathname, "training.json")

    if os.path.exists(filename):
        with open(filename) as data:
            training_dictionaries = json.load(data)
    else:
        training_dictionaries = None

    if training_dictionaries:
        for training_dictionary in training_dictionaries:
            training_dictionary["image"][
                "pathname"] = f"{pathname}{training_dictionary['image']['pathname']}"

            for instance in training_dictionary["objects"]:
                instance["mask"][
                    "pathname"] = f"{pathname}{instance['mask']['pathname']}"

    filename = os.path.join(pathname, "test.json")

    if os.path.exists(filename):
        with open(filename) as data:
            test_dictionaries = json.load(data)
    else:
        test_dictionaries = None

    if test_dictionaries:
        for test_dictionary in test_dictionaries:
            test_dictionary["image"][
                "pathname"] = f"{pathname}{training_dictionary['image']['pathname']}"

            for instance in test_dictionary["objects"]:
                instance["mask"][
                    "pathname"] = f"{pathname}{instance['mask']['pathname']}"

    return training_dictionaries, test_dictionaries
