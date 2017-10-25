import numpy

import keras_rcnn.preprocessing._object_detection
import keras_rcnn.datasets.malaria

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

class TestDebugObjectDetectionGenerator:
    def test_flow(self):
        generator = keras_rcnn.preprocessing.DebugObjectDetectionGenerator()

        classes = {
            "rbc": 1,
            "not":2
        }
        training, test = keras_rcnn.datasets.malaria.load_data()
        generator = generator.flow(training, classes)

        generator = keras_rcnn.preprocessing.ObjectDetectionGenerator()
        generator = generator.flow([training[1]], classes, target_shape=(1200, 1600), scale=1, ox=0, oy=0)