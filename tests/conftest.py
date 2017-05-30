import keras
import numpy
import pytest

import keras_rcnn.layers.object_detection


@pytest.fixture()
def anchor_layer():
    features = (14, 14)

    shape = (224, 224)

    return keras_rcnn.layers.object_detection.Anchor(features, shape)


@pytest.fixture()
def feat_h():
    return 14


@pytest.fixture()
def feat_w():
    return 14


@pytest.fixture()
def gt_boxes():
    boxes = [
        [24, 26, 75, 93, 0],
        [78, 14, 82, 23, 1],
        [78, 29, 27, 41, 2]
    ]

    return numpy.array(boxes, dtype=numpy.float)


@pytest.fixture()
def img_info():
    return numpy.array([224, 224], dtype=numpy.int32)


@pytest.fixture()
def object_proposal_layer():
    return keras_rcnn.layers.object_detection.ObjectProposal(300)


@pytest.fixture()
def object_proposal_model():
    a = keras.layers.Input((14, 14, 4 * 9))
    b = keras.layers.Input((14, 14, 2 * 9))

    y = keras_rcnn.layers.object_detection.ObjectProposal(300)([a, b])

    return keras.models.Model([a, b], y)


@pytest.fixture()
def x():
    return numpy.arange(2 * 9 * 14 * 14, dtype=numpy.float32).reshape(1, 18, 14, 14)
