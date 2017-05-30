import keras
import pytest

import keras_rcnn.layers.object_detection


@pytest.fixture()
def object_proposal():
    a = keras.layers.Input((14, 14, 4 * 9))
    b = keras.layers.Input((14, 14, 2 * 9))

    y = keras_rcnn.layers.object_detection.ObjectProposal(300)([a, b])

    return keras.models.Model([a, b], y)
