# -*- coding: utf-8 -*-

"""
Object detection
================

A simple example for ploting two figures of a exponential
function in order to test the autonomy of the gallery
stacking multiple images.
"""

import keras

import keras_rcnn.datasets.shape
import keras_rcnn.models
import keras_rcnn.preprocessing


def main():
    training_dictionary, test_dictionary = keras_rcnn.datasets.shape.load_data()

    categories = {"circle": 1, "rectangle": 2, "triangle": 3}

    generator = keras_rcnn.preprocessing.ObjectDetectionGenerator()

    generator = generator.flow_from_dictionary(
        dictionary=training_dictionary,
        categories=categories,
        target_size=(224, 224)
    )

    validation_data = keras_rcnn.preprocessing.ObjectDetectionGenerator()

    validation_data = validation_data.flow_from_dictionary(
        dictionary=test_dictionary,
        categories=categories,
        target_size=(224, 224)
    )

    keras.backend.set_learning_phase(1)

    model = keras_rcnn.models.RCNN(
        categories=["circle", "rectangle", "triangle"],
        dense_units=512,
        input_shape=(224, 224, 3)
    )

    optimizer = keras.optimizers.Adam()

    model.compile(optimizer)

    model.fit_generator(
        epochs=1,
        generator=generator,
        validation_data=validation_data
    )


if __name__ == '__main__':
    main()
