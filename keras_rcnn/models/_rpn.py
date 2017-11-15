# -*- coding: utf-8 -*-

import keras.backend
import keras.engine
import keras.layers

import keras_rcnn.backend
import keras_rcnn.classifiers
import keras_rcnn.datasets.malaria
import keras_rcnn.layers
import keras_rcnn.preprocessing


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
    def __init__(self, image, classes):
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

        features = keras.layers.Conv2D(64, name="convolution_1_1", **options)(image)
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

        convolution_3x3 = keras.layers.Conv2D(512, name="convolution_3x3", **options)(features)

        deltas = keras.layers.Conv2D(9 * 4, (1, 1), activation="linear", kernel_initializer="zero", name="deltas")(convolution_3x3)
        scores = keras.layers.Conv2D(9 * 1, (1, 1), activation="sigmoid", kernel_initializer="uniform", name="scores")(convolution_3x3)

        anchors, rpn_labels, bounding_box_targets = keras_rcnn.layers.AnchorTarget()([scores, bounding_boxes, metadata])

        deltas, scores = keras_rcnn.layers.RPN()([deltas, bounding_box_targets, scores, rpn_labels])

        proposals = keras_rcnn.layers.ObjectProposal()([metadata, deltas, scores, anchors])

        proposals, labels_targets, bounding_box_targets = keras_rcnn.layers.ProposalTarget()([proposals, labels, bounding_boxes])

        outputs = [anchors, proposals, deltas, scores]

        super(RPN, self).__init__(inputs, outputs)

    def compile(self, optimizer, **kwargs):
        super(RPN, self).compile(optimizer, None)
