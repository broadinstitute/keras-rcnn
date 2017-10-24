import keras.backend
import keras.engine
import keras.layers

import keras_rcnn.backend
import keras_rcnn.classifiers
import keras_rcnn.datasets.malaria
import keras_rcnn.layers
import keras_rcnn.preprocessing


def _extract_features(options=None, training=None):
    if options is None:
        options = {}

    def f(inputs):
        features = keras.layers.Conv2D(64, name="convolution_1_1", **options)(inputs)
        features = keras.layers.Conv2D(64, name="convolution_1_2", **options)(features)

        features = keras.layers.MaxPooling2D(strides=(2, 2), name="max_pooling_1")(features)

        features = keras.layers.Conv2D(128, name="convolution_2_1", **options)(features)
        features = keras.layers.Conv2D(128, name="convolution_2_2", **options)(features)

        features = keras.layers.MaxPooling2D(strides=(2, 2), name="max_pooling_2")(features)

        features = keras.layers.Conv2D(256, name="convolution_3_1", **options)(features)
        features = keras.layers.Conv2D(256, name="convolution_3_2", **options)(features)
        features = keras.layers.Conv2D(256, name="convolution_3_3", **options)(features)
        features = keras.layers.Conv2D(256, name="convolution_3_4", **options)(features)

        features = keras.layers.MaxPooling2D(strides=(2, 2), name="max_pooling_3")(features)

        features = keras.layers.Conv2D(512, name="convolution_4_1", **options)(features)
        features = keras.layers.Conv2D(512, name="convolution_4_2", **options)(features)
        features = keras.layers.Conv2D(512, name="convolution_4_3", **options)(features)
        features = keras.layers.Conv2D(512, name="convolution_4_4", **options)(features)

        features = keras.layers.MaxPooling2D(strides=(2, 2), name="max_pooling_4")(features)

        features = keras.layers.Conv2D(512, name="convolution_5_1", **options)(features)
        features = keras.layers.Conv2D(512, name="convolution_5_2", **options)(features)
        features = keras.layers.Conv2D(512, name="convolution_5_3", **options)(features)

        return features

    return f


def _extract_proposals(options=None):
    if options is None:
        options = {}

    def f(inputs):
        convolution_3x3 = keras.layers.Conv2D(512, name="convolution_3x3", **options)(inputs)

        deltas = keras.layers.Conv2D(9 * 4, (1, 1), activation="linear", kernel_initializer="zero", name="deltas")(convolution_3x3)
        scores = keras.layers.Conv2D(9 * 2, (1, 1), kernel_initializer="uniform", name="scores")(convolution_3x3)

        return [deltas, scores]

    return f


def _extract_regions(classes):
    def f(inputs):
        features, metadata, proposals = inputs

        regions = keras_rcnn.layers.RegionOfInterest(extent=(14, 14))([features, proposals, metadata])

        regions = keras.layers.TimeDistributed(keras.layers.Flatten())(regions)

        regions = keras.layers.TimeDistributed(keras.layers.Dense(256, activation="relu"))(regions)
        regions = keras.layers.TimeDistributed(keras.layers.Dropout(0.5))(regions)

        regions = keras.layers.TimeDistributed(keras.layers.Dense(256, activation="relu"))(regions)
        regions = keras.layers.TimeDistributed(keras.layers.Dropout(0.5))(regions)

        deltas = keras.layers.TimeDistributed(keras.layers.Dense(4 * classes, activation="linear", kernel_initializer="zero"))(regions)
        scores = keras.layers.TimeDistributed(keras.layers.Dense(classes, activation="softmax", kernel_initializer="zero"))(regions)

        return [deltas, scores]

    return f


def _train(classes):
    def f(inputs):
        options = {
            "activation": "relu",
            "kernel_size": (3, 3),
            "padding": "same"
        }

        bounding_boxes, image, labels, metadata = inputs

        features = _extract_features(options)(image)

        deltas, scores = _extract_proposals(options)(features)

        all_anchors, rpn_labels, bounding_box_targets = keras_rcnn.layers.AnchorTarget()([scores, bounding_boxes, metadata])

        scores_reshaped = keras.layers.Reshape((-1, 2))(scores)
        scores_reshaped = keras.layers.Activation("softmax")(scores_reshaped)

        deltas = keras_rcnn.layers.RPNRegressionLoss(9)([deltas, bounding_box_targets, rpn_labels])
        scores = keras_rcnn.layers.RPNClassificationLoss(9)([scores_reshaped, rpn_labels])

        proposals_ = keras_rcnn.layers.ObjectProposal()([metadata, deltas, scores, all_anchors])

        proposals, labels_targets, bounding_box_targets = keras_rcnn.layers.ProposalTarget()([proposals_, labels, bounding_boxes])

        deltas, scores = _extract_regions(classes)([features, metadata, proposals])

        deltas = keras_rcnn.layers.losses.RCNNRegressionLoss()([deltas, bounding_box_targets, labels_targets])
        scores = keras_rcnn.layers.losses.RCNNClassificationLoss()([scores, labels_targets])

        return [all_anchors, deltas, proposals, scores]

    return f
