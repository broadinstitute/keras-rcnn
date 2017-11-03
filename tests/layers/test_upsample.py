# -*- coding: utf-8 -*-

import keras
import numpy

import keras_rcnn.layers


class TestUpsample(object):
    def test_call(self):
        upsample = keras_rcnn.layers.Upsample()

        a = numpy.zeros((1, 2, 2, 1))
        b = numpy.zeros((1, 5, 5, 1))

        output = upsample.call([a, b])
        target = b

        numpy.testing.assert_array_equal(keras.backend.eval(output), target)
