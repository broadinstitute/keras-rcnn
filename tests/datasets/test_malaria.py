import keras_rcnn.datasets.malaria

def test_load_data():
    training, test = malaria.load_data()
    assert len(training) == 1208
    assert len(test) == 120