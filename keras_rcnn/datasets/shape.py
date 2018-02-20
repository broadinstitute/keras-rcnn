# -*- coding: utf-8 -*-

import json

import pkg_resources

import jsonschema


def load_data():
    resource_path = "/".join(["data", "schema.json"])

    with pkg_resources.resource_stream("keras_rcnn", resource_path) as stream:
        schema = json.load(stream)

    resource_path = "/".join(["data", "shape", "training.json"])

    with pkg_resources.resource_stream("keras_rcnn", resource_path) as stream:
        training_dictionary = json.load(stream)

    jsonschema.validate(training_dictionary, schema)

    resource_path = "/".join(["data", "shape", "test.json"])

    with pkg_resources.resource_stream("keras_rcnn", resource_path) as stream:
        test_dictionary = json.load(stream)

    jsonschema.validate(test_dictionary, schema)

    return training_dictionary, test_dictionary
