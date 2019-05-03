# -*- coding: utf-8 -*-

import keras.utils.data_utils
import numpy

import keras_rcnn.models


class JHung2019(keras_rcnn.models.RCNN):
    def compile(self, optimizer, **kwargs):
        super(JHung2019, self).compile(optimizer)

        origin = "http://keras-rcnn-applications.storage.googleapis.com/JHung2019.tar.gz"

        pathname = keras.utils.data_utils.get_file(
            cache_subdir='models',
            fname="JHung2019.hdf5",
            origin=origin,
            untar=True,
        )

        self.load_weights(pathname, by_name=True)

    def predict(self, x, batch_size=None, verbose=0, steps=None):
        prediction = super(JHung2019, self).predict(
            x,
            batch_size,
            verbose,
            steps
        )

        predicted_bounding_boxes, predicted_categories = prediction

        predicted_bounding_boxes = numpy.squeeze(predicted_bounding_boxes)

        predicted_categories = numpy.squeeze(predicted_categories)

        predicted_categories = numpy.argmax(predicted_categories, axis=-1)

        indices = numpy.where(predicted_categories > 0)

        predicted_bounding_boxes = predicted_bounding_boxes[indices]

        predicted_categories = predicted_categories[indices]

        return predicted_bounding_boxes, predicted_categories
