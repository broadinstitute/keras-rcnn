import keras
import numpy

import keras_rcnn.applications


class TestJHung2019:
    def setup(self):
        categories = {
            "red blood cell": 1,
            "leukocyte": 2,
            "ring": 3,
            "trophozoite": 4,
            "schizont": 5,
            "gametocyte": 6
        }

        self.instance = keras_rcnn.applications.JHung2019(
            (224, 224, 3),
            categories
        )

    def test_compile(self):
        optimizer = keras.optimizers.SGD()

        self.instance.compile(optimizer)

    def test_predict(self):
        image = numpy.random.random((1, 224, 224, 3))

        self.instance.predict(image)
