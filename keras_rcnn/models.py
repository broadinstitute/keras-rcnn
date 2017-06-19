# -*- coding: utf-8 -*-

import keras
import keras_rcnn.layers


class RCNN(keras.models.Model):
    def __init__(self, inputs, classes, regions_of_interest):
        y = keras_rcnn.layers.ROI(14, regions_of_interest)(inputs)

        y = keras.layers.Dense(4096)(y)
        y = keras.layers.TimeDistributed(y)

        score = keras.layers.Dense(classes)(y)
        score = keras.layers.Activation("softmax")(score)
        score = keras.layers.TimeDistributed(score)

        boxes = keras.layers.Dense(4 * (classes - 1))(y)
        boxes = keras.layers.Activation("linear")(boxes)
        boxes = keras.layers.TimeDistributed(boxes)

        super(RCNN, self).__init__(inputs, [score, boxes])


class RPN(keras.models.Model):
    def __init__(self, inputs):
        y = inputs.layers[-2].output

        a = keras.layers.Conv2D(9 * 1, (1, 1), activation="sigmoid")(y)
        b = keras.layers.Conv2D(9 * 4, (1, 1))(y)

        y = keras.layers.concatenate([a, b])

        super(RPN, self).__init__(inputs, y)
