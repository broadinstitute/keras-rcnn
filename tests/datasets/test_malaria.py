import keras_rcnn.datasets.malaria

def test_load_data():
    training, test = keras_rcnn.datasets.malaria.load_data()

    assert len(training) == 1208
    assert len(test) == 120