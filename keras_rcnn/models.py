import keras.backend
import keras.engine
import keras.layers

import keras_rcnn.backend
import keras_rcnn.classifiers
import keras_rcnn.datasets.malaria
import keras_rcnn.layers
import keras_rcnn.preprocessing


def _features(options=None):
    if options is None:
        options = {}

    def f(inputs):
        y = keras.layers.Conv2D(64, name="convolution_1_1", **options)(inputs)
        y = keras.layers.Conv2D(64, name="convolution_1_2", **options)(y)

        y = keras.layers.MaxPooling2D(strides=(2, 2), name="max_pooling_1")(y)

        y = keras.layers.Conv2D(128, name="convolution_2_1", **options)(y)
        y = keras.layers.Conv2D(128, name="convolution_2_2", **options)(y)

        y = keras.layers.MaxPooling2D(strides=(2, 2), name="max_pooling_2")(y)

        y = keras.layers.Conv2D(256, name="convolution_3_1", **options)(y)
        y = keras.layers.Conv2D(256, name="convolution_3_2", **options)(y)
        y = keras.layers.Conv2D(256, name="convolution_3_3", **options)(y)

        y = keras.layers.MaxPooling2D(strides=(2, 2), name="max_pooling_3")(y)

        y = keras.layers.Conv2D(512, name="convolution_4_1", **options)(y)
        y = keras.layers.Conv2D(512, name="convolution_4_2", **options)(y)
        y = keras.layers.Conv2D(512, name="convolution_4_3", **options)(y)

        y = keras.layers.MaxPooling2D(strides=(2, 2), name="max_pooling_4")(y)

        y = keras.layers.Conv2D(512, name="convolution_5_1", **options)(y)
        y = keras.layers.Conv2D(512, name="convolution_5_2", **options)(y)
        y = keras.layers.Conv2D(512, name="convolution_5_3", **options)(y)

        return y

    return f


def _proposals(options=None):
    if options is None:
        options = {}

    def f(inputs):
        boxes, features, labels, metadata = inputs

        y = keras.layers.Conv2D(512, name="convolution_3x3", **options)(features)

        deltas = keras.layers.Conv2D(9 * 4, (1, 1), name="deltas")(y)
        scores = keras.layers.Conv2D(9 * 2, (1, 1), name="scores")(y)

        anchors, rpn_labels, bounding_box_targets = keras_rcnn.layers.AnchorTarget()([scores, boxes, metadata])

        scores_reshaped = keras.layers.Reshape((-1, 2))(scores)
        scores_reshaped = keras.layers.Activation("softmax")(scores_reshaped)

        deltas = keras_rcnn.layers.losses.RPNRegressionLoss(9)([deltas, bounding_box_targets, rpn_labels])
        scores = keras_rcnn.layers.losses.RPNClassificationLoss(9)([scores_reshaped, rpn_labels])

        proposals_ = keras_rcnn.layers.ObjectProposal()([metadata, deltas, scores, anchors])

        proposals, label_targets, bounding_box_targets = keras_rcnn.layers.ProposalTarget()([proposals_, labels, boxes])

        return [proposals, label_targets, bounding_box_targets]

    return f


def _detections():
    def f(inputs):
        features, metadata, proposals, label_targets, bounding_box_targets = inputs

        yr = keras_rcnn.layers.RegionOfInterest()([features, proposals, metadata])

        yr = keras.layers.TimeDistributed(keras.layers.pooling.MaxPooling2D(pool_size=(2, 2)))(yr)

        y = keras.layers.TimeDistributed(keras.layers.Flatten())(yr)

        # y = keras.layers.TimeDistributed(keras.layers.Dense(4096, activation="relu"))(y)
        # y = keras.layers.TimeDistributed(keras.layers.Dense(4096, activation="relu"))(y)

        deltas = keras.layers.TimeDistributed(keras.layers.Dense(4 * 2, activation="linear"))(y)
        scores = keras.layers.TimeDistributed(keras.layers.Dense(1 * 2, activation="softmax"))(y)

        deltas = keras_rcnn.layers.losses.RCNNRegressionLoss()([deltas, bounding_box_targets, label_targets])
        scores = keras_rcnn.layers.losses.RCNNClassificationLoss()([scores, label_targets])

        return [deltas, scores]

    return f


def detections():
    def f(inputs):
        options = {
            "activation": "relu",
            "kernel_size": (3, 3),
            "padding": "same"
        }

        image, metadata, boxes, labels = inputs

        features = _features(options)(image)

        proposals = _proposals(options)([boxes, features, labels, metadata])

        detections = _detections()([features, metadata] + proposals)

        return detections

    return f
