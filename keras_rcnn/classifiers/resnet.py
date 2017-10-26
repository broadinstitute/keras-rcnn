# -*- coding: utf-8 -*-

import keras.backend
import keras.layers

import keras_resnet.blocks


def residual(classes, mask=False, features=512):
    """Resnet classifiers as in Mask R-CNN."""

    def f(x):
        if keras.backend.image_data_format() == "channels_last":
            channel_axis = 3
        else:
            channel_axis = 1

        # conv5 block as in Deep Residual Networks with first conv operates
        # on a 7x7 RoI with stride 1 (instead of 14x14 / stride 2)
        y = keras_resnet.blocks.time_distributed_bottleneck_2d(features, stage=3, block=0, stride=1)(x)
        y = keras_resnet.blocks.time_distributed_bottleneck_2d(features, stage=3, block=1, stride=1)(y)
        y = keras_resnet.blocks.time_distributed_bottleneck_2d(features, stage=3, block=2, stride=1)(y)

        # class and box branches
        y = keras.layers.TimeDistributed(keras.layers.GlobalAveragePooling2D())(y)

        score = keras.layers.TimeDistributed(keras.layers.Dense(classes, activation="softmax"))(y)
        boxes = keras.layers.TimeDistributed(keras.layers.Dense(4 * classes))(y)

        # TODO{JihongJu} the mask branch

        return [score, boxes]

    return f
