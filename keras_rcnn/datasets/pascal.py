# -*- coding: utf-8 -*-

import keras_rcnn.datasets


def load_data():
    """
    Load Everingham, et al.’s PASCAL Visual Object Classes (VOC) dataset.
    """
    return keras_rcnn.datasets.load_data("pascal")
