import json
import os.path

import keras.utils.data_utils


def load_data():
    """
    Load Hung, et al.’s malaria dataset.

    Hung, et al.’s malaria dataset is a collection of blood smears from multiple 
    patients captured by a variety of microscopes. Images are accompanied by 
    bounding boxes that capture each cell’s location and corresponding class labels 
    that describe each cell’s phenotype.
    """
    origin = "http://keras-rcnn.storage.googleapis.com/malaria.tar.gz"

    pathname = keras.utils.data_utils.get_file("malaria", origin=origin, untar=True)

    filename = os.path.join(pathname, "training.json")

    with open(filename) as data:
        training = json.load(data)

    filename = os.path.join(pathname, "test.json")

    with open(filename) as data:
        test = json.load(data)

    pathname = os.path.join(pathname, 'images')
    
    for dictionary in training:
        dictionary["filename"] = os.path.join(pathname, dictionary["filename"])

    for dictionary in test:
        dictionary["filename"] = os.path.join(pathname, dictionary["filename"])

    return training, test
