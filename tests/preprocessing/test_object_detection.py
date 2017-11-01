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

        image = keras.layers.Input((None, None, 3))
        training_options = {
            "anchor_target": {
                "allowed_border": 0,
                "clobber_positives": False,
                "negative_overlap": 0.3,
                "positive_overlap": 0.7,
            },
            "object_proposal": {
                "maximum_proposals": 500,
                "minimum_size": 16,
                "stride": 16
            },
            "proposal_target": {
                "fg_fraction": 0.2,
                "fg_thresh": 0.7,
                "bg_thresh_hi": 0.5,
                "bg_thresh_lo": 0.1,
            }
        }

        model = keras_rcnn.models.RCNN(image, classes=3, training_options=training_options)

        classes = {
            "rbc": 1,
            "not":2
        }

        generator = keras_rcnn.preprocessing.ObjectDetectionGenerator()
        training, test = keras_rcnn.datasets.malaria.load_data()
        generator = generator.flow(training, classes, target_shape=(448, 448), scale=1)

        optimizer = keras.optimizers.Adam(0.0001)

        model.compile(optimizer)

        model.fit_generator(generator, 1, epochs=1)
