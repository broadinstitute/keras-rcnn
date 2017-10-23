import keras
import numpy
import pytest

import keras_rcnn.layers.object_detection


@pytest.fixture()
def convolution_neural_network():
    options = {
        "activation": "relu",
        "kernel_size": (3, 3),
        "padding": "same"
    }

    shape = (224, 224, 3)

    x = keras.layers.Input(shape)

    y = keras.layers.Conv2D(64, **options)(x)
    y = keras.layers.Conv2D(64, **options)(y)

    y = keras.layers.MaxPooling2D(strides=(2, 2))(y)

    y = keras.layers.Conv2D(128, **options)(y)
    y = keras.layers.Conv2D(128, **options)(y)

    y = keras.layers.MaxPooling2D(strides=(2, 2))(y)

    y = keras.layers.Conv2D(256, **options)(y)
    y = keras.layers.Conv2D(256, **options)(y)
    y = keras.layers.Conv2D(256, **options)(y)
    y = keras.layers.Conv2D(256, **options)(y)

    y = keras.layers.MaxPooling2D(strides=(2, 2))(y)

    y = keras.layers.Conv2D(512, **options)(y)
    y = keras.layers.Conv2D(512, **options)(y)
    y = keras.layers.Conv2D(512, **options)(y)
    y = keras.layers.Conv2D(512, **options)(y)

    y = keras.layers.MaxPooling2D(strides=(2, 2))(y)

    y = keras.layers.Conv2D(512, **options)(y)
    y = keras.layers.Conv2D(512, **options)(y)
    y = keras.layers.Conv2D(512, **options)(y)

    return keras.models.Model(x, y)


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
def image_features():
    features = keras.layers.Input((14, 14, 3))

    return keras.layers.Conv2D(512, (1, 1), activation="relu")(features)


@pytest.fixture()
def image_features_npy():
    return numpy.random.random((1, 14, 14, 512))


@pytest.fixture()
def img_info():
    return numpy.array([224, 224], dtype=numpy.int32)


@pytest.fixture()
def pooling():
    return keras_rcnn.layers.RegionOfInterest([7, 7], 1)


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
def region_proposal_network():
    options = {
        "activation": "relu",
        "kernel_size": (3, 3),
        "padding": "same"
    }

    shape = (224, 224, 3)

    x = keras.layers.Input(shape)

    y = keras.layers.Conv2D(64, **options)(x)
    y = keras.layers.Conv2D(64, **options)(y)

    y = keras.layers.MaxPooling2D(strides=(2, 2))(y)

    y = keras.layers.Conv2D(128, **options)(y)
    y = keras.layers.Conv2D(128, **options)(y)

    y = keras.layers.MaxPooling2D(strides=(2, 2))(y)

    y = keras.layers.Conv2D(256, **options)(y)
    y = keras.layers.Conv2D(256, **options)(y)
    y = keras.layers.Conv2D(256, **options)(y)
    y = keras.layers.Conv2D(256, **options)(y)

    y = keras.layers.MaxPooling2D(strides=(2, 2))(y)

    y = keras.layers.Conv2D(512, **options)(y)
    y = keras.layers.Conv2D(512, **options)(y)
    y = keras.layers.Conv2D(512, **options)(y)
    y = keras.layers.Conv2D(512, **options)(y)

    y = keras.layers.MaxPooling2D(strides=(2, 2))(y)

    y = keras.layers.Conv2D(512, **options)(y)
    y = keras.layers.Conv2D(512, **options)(y)
    y = keras.layers.Conv2D(512, **options)(y)

    y = keras.layers.Conv2D(512, **options)(y)

    a = keras.layers.Conv2D(9 * 1, (1, 1), activation="sigmoid")(y)

    b = keras.layers.Conv2D(9 * 4, (1, 1))(y)

    y = keras_rcnn.layers.object_detection.ObjectProposal(300)([a, b])

    model = keras.models.Model(x, y)

    model.compile("sgd", "mse")


@pytest.fixture()
def x():
    x = numpy.arange(2 * 9 * 14 * 14, dtype=numpy.float32)
    return x.reshape(1, 18, 14, 14)


@pytest.fixture()
def regional_proposal_network_y_pred():
    return numpy.random.random((1, 4 * 9, 14, 14))
