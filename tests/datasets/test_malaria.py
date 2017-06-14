import os.path

import keras_rcnn.datasets.malaria


def test_load_data():
    training, test = keras_rcnn.datasets.malaria.load_data()

    assert os.path.exists(training[0]['filename'])

    assert os.path.exists(test[0]['filename'])

    assert len(training) == 1208

    assert len(test) == 120
