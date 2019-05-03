# -*- coding: utf-8 -*-

import keras.utils.data_utils

import keras_rcnn.models


class Hollandi2019(keras_rcnn.models.MaskRCNN):
    def compile(self, optimizer, **kwargs):
        super(Hollandi2019, self).compile(optimizer)

        origin = "http://keras-rcnn.storage.googleapis.com/Hollandi2019.tar.gz"

        pathname = keras.utils.data_utils.get_file(
            cache_subdir='models',
            fname="Hollandi2019",
            origin=origin,
            untar=True,
        )

        self.load_weights(pathname, by_name=True)

    def predict(self, x, batch_size=None, verbose=0, steps=None):
        super(Hollandi2019, self).predict(x, batch_size, verbose, steps)
