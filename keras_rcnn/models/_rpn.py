# -*- coding: utf-8 -*-

import keras.backend
import keras.engine
import keras.layers
import keras_resnet.models

import keras_rcnn.backend
import keras_rcnn.classifiers
import keras_rcnn.datasets.malaria
import keras_rcnn.layers
import keras_rcnn.preprocessing


def pyramid(c3, c4, c5, features):
    p4 = keras.layers.Conv2D(features, kernel_size=1, strides=1, padding='same', name='p4')(c5)

    upsampled_p4 = keras_rcnn.layers.Upsample(name='upsampled_p4')([p4, c4])

    p3 = keras.layers.Conv2D(features, kernel_size=1, strides=1, padding='same', name='C4_reduced')(c4)
    p3 = keras.layers.Add(name='p5_merged')([upsampled_p4, p3])
    p3 = keras.layers.Conv2D(features, kernel_size=3, strides=1, padding='same', name='p3')(p3)

    upsampled_p3 = keras_rcnn.layers.Upsample(name='upsampled_p3')([p3, c3])

    p2 = keras.layers.Conv2D(features, kernel_size=1, strides=1, padding='same', name='C3_reduced')(c3)
    p2 = keras.layers.Add(name='p4_merged')([upsampled_p3, p2])
    p2 = keras.layers.Conv2D(features, kernel_size=3, strides=1, padding='same', name='p2')(p2)

    p5 = keras.layers.Conv2D(features, kernel_size=3, strides=2, padding='same', name='p5')(c5)

    p6 = keras.layers.Activation('relu', name='C6_relu')(p5)
    p6 = keras.layers.Conv2D(features, kernel_size=3, strides=2, padding='same', name='p6')(p6)

    return p2, p3, p4, p5, p6


class RPN(keras.models.Model):
    """
    Example:

    Load data and create object detection generators:

            training, test = keras_rcnn.datasets.malaria.load_data()

            training, validation = sklearn.model_selection.train_test_split(training)

            classes = {"rbc": 1, "not":2}

            generator = keras_rcnn.preprocessing.ObjectDetectionGenerator()

            generator = generator.flow(training, classes, (448, 448), 1.0)

            validation_data = keras_rcnn.preprocessing.ObjectDetectionGenerator()

            validation_data = validation_data.flow(validation, classes, (448, 448), 1.0)

    Create an instance of the RPN model:

            image = keras.layers.Input((448, 448, 3))

            model = RPN(image, classes=len(classes) + 1)

            optimizer = keras.optimizers.Adam(0.0001)

            model.compile(optimizer)

    Train the model:

            model.fit_generator(
                epochs=10,
                generator=generator,
                steps_per_epoch=1000
            )

    Predict and visualize your anchors or proposals:

            example, _ = generator.next()

            target_bounding_boxes, target_image, target_labels, _ = example

            target_bounding_boxes = numpy.squeeze(target_bounding_boxes)

            target_image = numpy.squeeze(target_image)

            target_labels = numpy.argmax(target_labels, -1)

            target_labels = numpy.squeeze(target_labels)

            output_anchors, output_proposals, output_deltas, output_scores = model.predict(example)

            output_anchors = numpy.squeeze(output_anchors)

            output_proposals = numpy.squeeze(output_proposals)

            output_deltas = numpy.squeeze(output_deltas)

            output_scores = numpy.squeeze(output_scores)

            _, axis = matplotlib.pyplot.subplots(1)

            axis.imshow(target_image)

            for index, label in enumerate(target_labels):
                if label == 1:
                    xy = [
                        target_bounding_boxes[index][0],
                        target_bounding_boxes[index][1]
                    ]

                    w = target_bounding_boxes[index][2] - target_bounding_boxes[index][0]
                    h = target_bounding_boxes[index][3] - target_bounding_boxes[index][1]

                    rectangle = matplotlib.patches.Rectangle(xy, w, h, edgecolor="g", facecolor="none")

                    axis.add_patch(rectangle)

            for index, score in enumerate(output_scores):
                if score > 0.95:
                    xy = [
                        output_anchors[index][0],
                        output_anchors[index][1]
                    ]

                    w = output_anchors[index][2] - output_anchors[index][0]
                    h = output_anchors[index][3] - output_anchors[index][1]

                    rectangle = matplotlib.patches.Rectangle(xy, w, h, edgecolor="r", facecolor="none")

                    axis.add_patch(rectangle)

            matplotlib.pyplot.show()
    """
    def __init__(self, image, classes, feature_maps=None, features=256):
        if feature_maps is None:
            feature_maps = [32, 64, 128, 256, 512]

        inputs = [
            keras.layers.Input((None, 4)),
            image,
            keras.layers.Input((None, classes)),
            keras.layers.Input((3,))
        ]

        options = {
            "activation": "relu",
            "kernel_size": (3, 3),
            "padding": "same"
        }

        bounding_boxes, image, labels, metadata = inputs

        _, c3, c4, c5 = keras_resnet.models.ResNet50(image, include_top=False).outputs

        p2, p3, p4, p5, p6 = pyramid(c3, c4, c5, features)

        number_of_anchors = len(feature_maps) * 3

        pyramidal_deltas = []
        pyramidal_scores = []

        pyramidal_target_anchors = []

        pyramidal_target_bounding_boxes = []

        pyramidal_target_scores = []

        for index, feature_map in enumerate(feature_maps):
            name = f"p{index + 2}"

            convolution_3x3 = keras.layers.Conv2D(features, **options)(locals()[name])

            deltas = keras.layers.Conv2D(number_of_anchors * 4, (1, 1), activation="linear", kernel_initializer="zero")(convolution_3x3)
            scores = keras.layers.Conv2D(number_of_anchors * 1, (1, 1), activation="sigmoid", kernel_initializer="uniform")(convolution_3x3)

            target_anchors, target_scores, target_bounding_boxes = keras_rcnn.layers.AnchorTarget(base_size=(feature_map - 1), scales=[1])([scores, bounding_boxes, metadata])

            deltas = keras.layers.Reshape((-1, 4))(deltas)
            scores = keras.layers.Reshape((-1,))(scores)

            pyramidal_deltas.append(deltas)
            pyramidal_scores.append(scores)

            pyramidal_target_anchors.append(target_anchors)

            pyramidal_target_bounding_boxes.append(target_bounding_boxes)

            pyramidal_target_scores.append(target_scores)

        anchors = keras.layers.concatenate(pyramidal_target_anchors, 1)
        deltas = keras.layers.concatenate(pyramidal_deltas, 1)
        scores = keras.layers.concatenate(pyramidal_scores, 1)
        rpn_labels = keras.layers.concatenate(pyramidal_target_scores, 1)
        bounding_box_targets = keras.layers.concatenate(pyramidal_target_bounding_boxes, 1)

        deltas, scores = keras_rcnn.layers.RPN()([deltas, bounding_box_targets, scores, rpn_labels])

        proposals = keras_rcnn.layers.ObjectProposal()([metadata, deltas, scores, anchors])

        proposals, labels_targets, bounding_box_targets = keras_rcnn.layers.ProposalTarget()([proposals, labels, bounding_boxes])

        outputs = [anchors, proposals, deltas, scores]

        super(RPN, self).__init__(inputs, outputs)

    def compile(self, optimizer, **kwargs):
        super(RPN, self).compile(optimizer, None)
