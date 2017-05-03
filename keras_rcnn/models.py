# -*- coding: utf-8 -*-

import keras.layers
import keras.models
import keras.layers
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
        boxes = keras.layers.Activation("linear")(score)
        boxes = keras.layers.TimeDistributed(score)

        super(RCNN, self).__init__(inputs, [score, boxes])


class RPN(keras.models.Model):
    def __init__(self, inputs, anchors):
        y = keras.layers.Conv2D(512, (3, 3), padding="same")(inputs)
        y = keras.layers.Activation("relu")(y)

        score = keras.layers.Conv2D(anchors, (1, 1))(y)
        score = keras.layers.Activation("sigmoid")(score)

        boxes = keras.layers.Conv2D(anchors * 4, (1, 1))(y)
        boxes = keras.layers.Activation("linear")(boxes)

        super(RPN, self).__init__(inputs, [score, boxes])
