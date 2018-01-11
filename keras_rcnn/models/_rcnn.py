# -*- coding: utf-8 -*-

import keras.backend
import keras.engine
import keras.layers

import keras_rcnn.backend
import keras_rcnn.classifiers
import keras_rcnn.datasets.malaria
import keras_rcnn.layers
import keras_rcnn.preprocessing


class RCNN(keras.models.Model):
    def __init__(self, target_image, classes):
        inputs = [
            keras.layers.Input((None, 4)),
            target_image,
            keras.layers.Input((None, classes)),
            keras.layers.Input((3,))
        ]

        options = {
            "activation": "relu",
            "kernel_size": (3, 3),
            "padding": "same"
        }

        target_bounding_boxes, target_image, target_labels, target_metadata = inputs

        output_features = keras.layers.Conv2D(64, name='block1_conv1', **options)(target_image)
        output_features = keras.layers.Conv2D(64, name='block1_conv2', **options)(output_features)

        output_features = keras.layers.MaxPooling2D(strides=(2, 2), name='block1_pool')(output_features)

        output_features = keras.layers.Conv2D(128, name='block2_conv1', **options)(output_features)
        output_features = keras.layers.Conv2D(128, name='block2_conv2', **options)(output_features)

        output_features = keras.layers.MaxPooling2D(strides=(2, 2), name='block2_pool')(output_features)

        output_features = keras.layers.Conv2D(256, name='block3_conv1', **options)(output_features)
        output_features = keras.layers.Conv2D(256, name='block3_conv2', **options)(output_features)
        output_features = keras.layers.Conv2D(256, name='block3_conv3', **options)(output_features)
        output_features = keras.layers.Conv2D(256, name='block3_conv4', **options)(output_features)

        output_features = keras.layers.MaxPooling2D(strides=(2, 2), name='block3_pool')(output_features)

        output_features = keras.layers.Conv2D(512, name='block4_conv1', **options)(output_features)
        output_features = keras.layers.Conv2D(512, name='block4_conv2', **options)(output_features)
        output_features = keras.layers.Conv2D(512, name='block4_conv3', **options)(output_features)
        output_features = keras.layers.Conv2D(512, name='block4_conv4', **options)(output_features)

        output_features = keras.layers.MaxPooling2D(strides=(2, 2), name='block4_pool')(output_features)

        output_features = keras.layers.Conv2D(512, name='block5_conv1', **options)(output_features)
        output_features = keras.layers.Conv2D(512, name='block5_conv2', **options)(output_features)
        output_features = keras.layers.Conv2D(512, name='block5_conv3', **options)(output_features)

        convolution_3x3 = keras.layers.Conv2D(512, name="convolution_3x3", **options)(output_features)

        output_deltas = keras.layers.Conv2D(9 * 4, (1, 1), activation="linear", kernel_initializer="zero", name="deltas")(convolution_3x3)
        output_scores = keras.layers.Conv2D(9 * 1, (1, 1), activation="sigmoid", kernel_initializer="uniform", name="scores")(convolution_3x3)

        target_anchors, target_proposal_labels, target_proposals = keras_rcnn.layers.AnchorTarget()([output_scores, target_bounding_boxes, target_metadata])

        output_deltas, output_scores = keras_rcnn.layers.RPN()([output_deltas, target_proposals, output_scores, target_proposal_labels])

        output_proposals = keras_rcnn.layers.ObjectProposal()([target_metadata, output_deltas, output_scores, target_anchors])

        output_proposals, target_proposal_labels, target_proposals = keras_rcnn.layers.ProposalTarget()([output_proposals, target_labels, target_bounding_boxes])

        output_regions = keras_rcnn.layers.RegionOfInterest(extent=(14, 14))([output_features, output_proposals, target_metadata])

        output_regions = keras.layers.TimeDistributed(keras.layers.Flatten())(output_regions)

        output_regions = keras.layers.TimeDistributed(keras.layers.Dense(256, activation="relu"))(output_regions)
        output_regions = keras.layers.TimeDistributed(keras.layers.Dense(256, activation="relu"))(output_regions)

        output_deltas = keras.layers.TimeDistributed(keras.layers.Dense(4 * classes, activation="linear", kernel_initializer="zero"))(output_regions)
        output_scores = keras.layers.TimeDistributed(keras.layers.Dense(1 * classes, activation="softmax", kernel_initializer="zero"))(output_regions)

        output_deltas, output_scores = keras_rcnn.layers.RCNN()([target_proposals, target_proposal_labels, output_deltas, output_scores])

        output_bounding_boxes, output_labels = keras_rcnn.layers.ObjectDetection()([output_proposals, output_deltas, output_scores, target_metadata])

        outputs = [output_bounding_boxes, output_labels]

        super(RCNN, self).__init__(inputs, outputs)

    def compile(self, optimizer, **kwargs):
        super(RCNN, self).compile(optimizer, None)
