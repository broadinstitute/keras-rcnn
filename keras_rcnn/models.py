# -*- coding: utf-8 -*-

import keras
import keras_resnet.models

import keras_rcnn.layers
import keras_rcnn.classifiers


class RCNN(keras.models.Model):
    """
    Faster R-CNN model by S Ren et, al. (2015).

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
    :param features: (Convolutional) features (e.g. extracted by `keras_resnet.models.ResNet50`)
    :param heads: R-CNN classifiers for object detection and/or segmentation on the proposed regions
    :param rois: integer, number of regions of interest per image

    :return model: a functional model API for R-CNN.
    """

    def __init__(self, inputs, features, heads, rois):
        num_anchors = 9 # TODO: Parametrize this

        # Propose regions given the features
        rpn_conv           = keras.layers.Conv2D(512            , kernel_size=(3, 3), padding="same", activation="relu", name="rpn_conv")(features)
        rpn_classification = keras.layers.Conv2D(num_anchors * 2, kernel_size=(1, 1), activation="sigmoid", name="rpn_cls_prob")(rpn_conv)
        rpn_regression     = keras.layers.Conv2D(num_anchors * 4, kernel_size=(1, 1), name="rpn_bbox_pred")(rpn_conv)

        rpn_prediction = keras.layers.concatenate([rpn_classification, rpn_regression])

        proposals = keras_rcnn.layers.object_detection.ObjectProposal(rois)([rpn_regression, rpn_classification])

        # Apply the classifiers on the proposed regions
        slices = keras_rcnn.layers.ROI((7, 7))([inputs, proposals])

        [score, boxes] = heads(slices)

        super(RCNN, self).__init__(inputs, [rpn_prediction, score, boxes])


class ResNet50RCNN(RCNN):
    """
    Faster R-CNN model with ResNet50.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
    :param classes: integer, number of classes
    :param rois: integer, number of regions of interest per image
    :param blocks: list of blocks to use in ResNet50

    :return model: a functional model API for R-CNN.
    """

    def __init__(self, inputs, classes, rois=300, blocks=[3, 4, 6]):
        # ResNet50 as encoder
        features = keras_resnet.models.ResNet50(inputs, blocks=blocks, include_top=False).output

        # ResHead with score and boxes
        heads = keras_rcnn.classifiers.residual(classes)

        super(ResNet50RCNN, self).__init__(inputs, features, heads, rois)
