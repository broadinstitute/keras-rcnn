import json
import os.path

import keras.utils.data_utils


def load_data():
    origin = "http://keras-rcnn.storage.googleapis.com/malaria.tar.gz"

    pathname = keras.utils.data_utils.get_file("malaria", origin=origin, untar=True)

    filename = os.path.join(pathname, "training.json")

    with open(filename) as data:
        training = json.load(data)

    filename = os.path.join(pathname, "test.json")

    with open(filename) as data:
        test = json.load(data)

    return training, test