# -*- coding: utf-8 -*-

import json
import os.path

import pkg_resources


def load_data():
    resource_path = os.path.join("data", "shape", "training.json")

    with pkg_resources.resource_stream("keras_rcnn", resource_path) as stream:
        training_dictionary = json.load(stream)

    resource_path = os.path.join("data", "shape", "test.json")

    with pkg_resources.resource_stream("keras_rcnn", resource_path) as stream:
        test_dictionary = json.load(stream)

    return training_dictionary, test_dictionary
