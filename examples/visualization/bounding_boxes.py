# -*- coding: utf-8 -*-

"""
Bounding boxes
==============

A simple example for ploting two figures of a exponential
function in order to test the autonomy of the gallery
stacking multiple images.
"""

import numpy
import keras_rcnn.datasets.shape
import keras_rcnn.preprocessing
import keras_rcnn.utils


def main():
    training_dictionary, test_dictionary = keras_rcnn.datasets.shape.load_data()

    categories = {"circle": 1, "rectangle": 2, "triangle": 3}

    generator = keras_rcnn.preprocessing.ObjectDetectionGenerator()

    generator = generator.flow_from_dictionary(
        dictionary=training_dictionary,
        categories=categories,
        target_size=(224, 224),
        shuffle=False
    )

    target, _ = generator.next()

    target_bounding_boxes, target_categories, target_images, target_masks, _ = target

    target_bounding_boxes = numpy.squeeze(target_bounding_boxes)

    target_images = numpy.squeeze(target_images)

    target_categories = numpy.argmax(target_categories, -1)

    target_categories = numpy.squeeze(target_categories)

    keras_rcnn.utils.show_bounding_boxes(target_images, target_bounding_boxes, target_categories)


if __name__ == '__main__':
    main()
