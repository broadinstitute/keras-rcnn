import pytest

import keras_rcnn.datasets.shape
import keras_rcnn.layers
import keras_rcnn.preprocessing


@pytest.fixture()
def anchor_layer():
    return keras_rcnn.layers.Anchor()


@pytest.fixture()
def generator(training_dictionary):
    object_detection_generator = keras_rcnn.preprocessing.ObjectDetectionGenerator()

    categories = {"circle": 1, "rectangle": 2, "triangle": 3}

    return object_detection_generator.flow_from_dictionary(
        dictionary=training_dictionary,
        categories=categories,
        target_size=(224, 224),
        shuffle=False
    )


@pytest.fixture()
def training_dictionary():
    return keras_rcnn.datasets.shape.load_data()[0]


@pytest.fixture()
def test_dictionary():
    return keras_rcnn.datasets.shape.load_data()[1]
