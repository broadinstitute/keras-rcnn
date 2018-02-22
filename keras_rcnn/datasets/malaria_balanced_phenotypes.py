# -*- coding: utf-8 -*-

import keras_rcnn.datasets


def load_data():
    """
    Load Hung, et al.’s malaria dataset.
    Hung, et al.’s malaria dataset is a collection of blood smears from
    multiple patients captured by a variety of microscopes. Images are
    accompanied by bounding boxes that capture each cell’s location and
    corresponding class labels that describe each cell’s phenotype.
    6 object classes plus "difficult": "rbc", "leu", "ring", "tro", "sch", "gam"
    These are 448x448 crops of the full sized images, crops containing
    non rbc cells were rotated to better balance the classes.
    """

    return keras_rcnn.datasets.load_data("malaria_balanced_phenotypes")
