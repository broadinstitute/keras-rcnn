# -*- coding: utf-8 -*-

import json

import jsonschema
import pkg_resources


def load_data():
    resource_path = "/".join(["data", "schema.json"])

    with pkg_resources.resource_stream("keras_rcnn", resource_path) as stream:
        schema = json.load(stream)

    resource_path = "/".join(["data", "shape", "training.json"])

    with pkg_resources.resource_stream("keras_rcnn", resource_path) as stream:
        training_dictionary = json.load(stream)

    jsonschema.validate(training_dictionary, schema)

    for dictionary in training_dictionary:
        resource_path = "/".join(["data", "shape", dictionary["image"]["pathname"]])

        pathname = pkg_resources.resource_filename("keras_rcnn", resource_path)

        dictionary["image"]["pathname"] = pathname

    resource_path = "/".join(["data", "shape", "test.json"])

    with pkg_resources.resource_stream("keras_rcnn", resource_path) as stream:
        test_dictionary = json.load(stream)

    jsonschema.validate(test_dictionary, schema)

    for dictionary in test_dictionary:
        resource_path = "/".join(["data", "shape", dictionary["image"]["pathname"]])

        pathname = pkg_resources.resource_filename("keras_rcnn", resource_path)

        dictionary["image"]["pathname"] = pathname

    return training_dictionary, test_dictionary
