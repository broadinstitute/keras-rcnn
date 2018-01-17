import numpy

import keras_rcnn.preprocessing._object_detection
import keras_rcnn.datasets.malaria
import keras_rcnn.models
import keras


def test_scale_shape():
    min_size = 200
    max_size = 300
    size = (600, 1000)

    size, scale = keras_rcnn.preprocessing._object_detection.scale_size(
        size, min_size, max_size
    )

    expected = (180, 300)
    numpy.testing.assert_equal(size, expected)

    expected = 0.3

    assert numpy.isclose(scale, expected)


class TestObjectDetectionGenerator:
    def test_flow(self):
        classes = {
            "rbc": 1,
            "not":2
        }

        training, _, _ = keras_rcnn.datasets.malaria.load_data()

        generator = keras_rcnn.preprocessing.ObjectDetectionGenerator()

        generator = generator.flow(training, classes, target_shape=(448, 448), scale=1)
