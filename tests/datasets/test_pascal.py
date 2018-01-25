import os.path

import keras_rcnn.datasets.pascal


def test_load_data():
    training, validation, test = keras_rcnn.datasets.pascal.load_data()

    assert os.path.exists(training[0]['filename'])

    assert os.path.exists(validation[0]['filename'])

    assert os.path.exists(test[0]['filename'])


