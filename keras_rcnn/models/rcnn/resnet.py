# -*- coding: utf-8 -*-

import keras
import keras_resnet.models
import keras_rcnn.models
import keras_rcnn.classifiers

"""
Faster R-CNN model with ResNet50.

:param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
:param classes: integer, number of classes
:param rois: integer, number of regions of interest per image
:param blocks: list of blocks to use in ResNet50

:return model: a functional model API for R-CNN.
"""

def ResNet50RCNN(inputs, classes, rois=300, blocks=[3, 4, 6], *args, **kwargs):
    # ResNet50 as encoder
    image, _, _ = inputs

    # Run encoder (ResNet50)
    features = keras_resnet.models.ResNet50(image, blocks=blocks, include_top=False, name="resnet50")(image)

    # Run RPN
    proposals, rpn_classification_loss, rpn_regression_loss = keras_rcnn.models.rpn(inputs, features, rois)

    # Perform ROI Pooling on the proposed regions
    slices = keras_rcnn.layers.ROI((7, 7))([image, proposals])

    # Run classifier to get scores and boxes
    scores, boxes = keras_rcnn.classifiers.residual(classes)(slices)

    return keras.models.Model(inputs=inputs, outputs=[scores, boxes, features, rpn_classification_loss, rpn_regression_loss], *args, **kwargs)
