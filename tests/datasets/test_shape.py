import keras_rcnn.datasets.shape


def test_load_data():
    training, test = keras_rcnn.datasets.shape.load_data()

    assert len(training) == 256

    assert len(test) == 256
