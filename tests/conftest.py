import pytest

import keras_rcnn.datasets.shape


@pytest.fixture()
def training_dictionary():
    return keras_rcnn.datasets.shape.load_data()[0]


@pytest.fixture()
def test_dictionary():
    return keras_rcnn.datasets.shape.load_data()[1]
