# -*- coding: utf-8 -*-

import keras
import keras_resnet.models


def ResNet50():
    def f(x):
        y = keras_resnet.models.ResNet50(
            include_top=False,
            inputs=x
        )

        _, _, _, convolution_5 = y.outputs

        return convolution_5

    return f


def VGG16():
    def f(x):
        y = keras.applications.VGG16(
            include_top=False,
            input_tensor=x
        )

        return y.layers[-3].output

    return f


def VGG19():
    def f(x):
        y = keras.applications.VGG19(
            include_top=False,
            input_tensor=x
        )

        return y.layers[-3].output

    return f
