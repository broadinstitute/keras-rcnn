import keras.layers
import numpy

import keras_rcnn.layers.object_detection


class TestObjectProposal:
    def test_call(self):
        a = keras.layers.Input((14, 14, 4 * 9))
        b = keras.layers.Input((14, 14, 2 * 9))

        y = keras_rcnn.layers.object_detection.ObjectProposal(300)([a, b])

        model = keras.models.Model([a, b], y)

        model.compile("sgd", "mse")

        a = numpy.random.rand(1, 14, 14, 4 * 9)
        b = numpy.random.rand(1, 14, 14, 2 * 9)

        prediction = model.predict([a, b])

        assert prediction.shape == (1, 100, 4)
