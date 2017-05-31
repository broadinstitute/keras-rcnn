import keras.layers
import keras.models
import numpy

import keras_rcnn.layers.object_detection


class TestObjectProposal:
    def test_build(self):
        assert True

    def test_call(self):
        options = {
            "activation": "relu",
            "kernel_size": (3, 3),
            "padding": "same"
        }

        shape = (224, 224, 3)

        x = keras.layers.Input(shape)

        y = keras.layers.Conv2D(64, **options)(x)
        y = keras.layers.Conv2D(64, **options)(y)

        y = keras.layers.MaxPooling2D(strides=(2, 2))(y)

        y = keras.layers.Conv2D(128, **options)(y)
        y = keras.layers.Conv2D(128, **options)(y)

        y = keras.layers.MaxPooling2D(strides=(2, 2))(y)

        y = keras.layers.Conv2D(256, **options)(y)
        y = keras.layers.Conv2D(256, **options)(y)
        y = keras.layers.Conv2D(256, **options)(y)
        y = keras.layers.Conv2D(256, **options)(y)

        y = keras.layers.MaxPooling2D(strides=(2, 2))(y)

        y = keras.layers.Conv2D(512, **options)(y)
        y = keras.layers.Conv2D(512, **options)(y)
        y = keras.layers.Conv2D(512, **options)(y)
        y = keras.layers.Conv2D(512, **options)(y)

        y = keras.layers.MaxPooling2D(strides=(2, 2))(y)

        y = keras.layers.Conv2D(512, **options)(y)
        y = keras.layers.Conv2D(512, **options)(y)
        y = keras.layers.Conv2D(512, **options)(y)

        y = keras.layers.Conv2D(512, **options)(y)

        a = keras.layers.Conv2D(9 * 4, (1, 1))(y)

        b = keras.layers.Conv2D(9 * 2, (1, 1), activation="sigmoid")(y)

        y = keras_rcnn.layers.object_detection.ObjectProposal(300)([a, b])

        model = keras.models.Model(x, y)

        model.compile("sgd", "mse")

        image = numpy.random.rand(1, 224, 224, 3)

        prediction = model.predict(image)

        assert prediction.shape == (1, 300, 4)

    def test_compute_output_shape(self, object_proposal_layer):
        assert object_proposal_layer.compute_output_shape((14, 14)) == (None, 300, 4)
