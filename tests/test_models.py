import keras.backend
import keras.layers
import keras.models
import numpy

import keras_rcnn.datasets.malaria
import keras_rcnn.layers
import keras_rcnn.models
import keras_rcnn.preprocessing


def test_vgg16_rcnn():
    image = keras.layers.Input((224, 224, 3), name="image")

    bounding_boxes = keras.layers.Input((None, 4), name="bounding_boxes")

    labels = keras.layers.Input((None, 2), name="labels")

    metadata = keras.layers.Input((3,), name="metadata")

    # CNN
    options = {
        "activation": "relu",
        "kernel_size": (3, 3),
        "padding": "same"
    }

    features = keras.layers.Conv2D(64, name="convolution_1_1", **options)(image)
    features = keras.layers.Conv2D(64, name="convolution_1_2", **options)(features)

    features = keras.layers.MaxPooling2D(strides=(2, 2), name="max_pooling_1")(features)

    features = keras.layers.Conv2D(128, name="convolution_2_1", **options)(features)
    features = keras.layers.Conv2D(128, name="convolution_2_2", **options)(features)

    features = keras.layers.MaxPooling2D(strides=(2, 2), name="max_pooling_2")(features)

    features = keras.layers.Conv2D(256, name="convolution_3_1", **options)(features)
    features = keras.layers.Conv2D(256, name="convolution_3_2", **options)(features)
    features = keras.layers.Conv2D(256, name="convolution_3_3", **options)(features)

    features = keras.layers.MaxPooling2D(strides=(2, 2), name="max_pooling_3")(features)

    features = keras.layers.Conv2D(512, name="convolution_4_1", **options)(features)
    features = keras.layers.Conv2D(512, name="convolution_4_2", **options)(features)
    features = keras.layers.Conv2D(512, name="convolution_4_3", **options)(features)

    features = keras.layers.MaxPooling2D(strides=(2, 2), name="max_pooling_4")(features)

    features = keras.layers.Conv2D(512, name="convolution_5_1", **options)(features)
    features = keras.layers.Conv2D(512, name="convolution_5_2", **options)(features)
    features = keras.layers.Conv2D(512, name="convolution_5_3", **options)(features)

    convolution_3x3 = keras.layers.Conv2D(512, name="convolution_3x3", **options)(features)

    deltas = keras.layers.Conv2D(9 * 4, (1, 1), name="deltas")(convolution_3x3)
    scores = keras.layers.Conv2D(9 * 2, (1, 1), name="scores", activation="sigmoid")(convolution_3x3)

    rpn_labels, bounding_box_targets = keras_rcnn.layers.AnchorTarget()([scores, bounding_boxes, metadata])

    deltas = keras_rcnn.layers.losses.RPNRegressionLoss(9)([deltas, bounding_box_targets, rpn_labels])
    scores = keras_rcnn.layers.losses.RPNClassificationLoss(9)([scores, rpn_labels])

    proposals = keras_rcnn.layers.ObjectProposal()([metadata, deltas, scores])

    proposals, labels_targets, bounding_box_targets = keras_rcnn.layers.ProposalTarget()([proposals, bounding_boxes, labels])

    y = keras_rcnn.layers.RegionOfInterest()([features, proposals])

    y = keras.layers.TimeDistributed(keras.layers.Flatten())(y)

    y = keras.layers.TimeDistributed(keras.layers.Dense(4096, activation="relu"))(y)
    y = keras.layers.TimeDistributed(keras.layers.Dense(4096, activation="relu"))(y)

    boxes = keras.layers.TimeDistributed(keras.layers.Dense(4*2, activation="linear"))(y)
    klass = keras.layers.TimeDistributed(keras.layers.Dense(2, activation="softmax"))(y)

    boxes = keras_rcnn.layers.losses.RCNNRegressionLoss()([bounding_box_targets, boxes])
    klass = keras_rcnn.layers.losses.RCNNClassificationLoss()([labels_targets, klass])

    model = keras.models.Model([image, bounding_boxes, metadata, labels], [boxes, klass])
    model.compile("adam", None)

    training, test = keras_rcnn.datasets.malaria.load_data()
    generator = keras_rcnn.preprocessing.ObjectDetectionGenerator()

    generator = generator.flow(training)
    x_image, x_boxes = generator.next()
    x_metadata = numpy.expand_dims([224, 224, 3], 0)
    x_labels = keras.utils.to_categorical(numpy.random.choice((0.0, 1.0), (x_boxes.shape[1],)))
    x_labels = numpy.expand_dims(x_labels, 0)

    model.fit([x_image, x_boxes, x_metadata, x_labels], [x_boxes, x_labels], epochs=10)
