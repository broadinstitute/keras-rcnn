# -*- coding: utf-8 -*-

import keras.backend
import keras.layers


class MaskRCNN(keras.layers.Layer):
    def __init__(self, threshold=0.5, **kwargs):
        self.threshold = threshold

        super(MaskRCNN, self).__init__(**kwargs)

    def call(self, inputs, training=None, **kwargs):
        target_bounding_boxes, output_bounding_boxes, target_masks, output_masks = inputs

        loss = self.compute_mask_loss(
            target_bounding_boxes,
            output_bounding_boxes,
            target_masks,
            output_masks
        )

        self.add_loss(loss, inputs)

        return output_masks

    @staticmethod
    def intersection_over_union(a, b):
        # TODO: Write `split` Keras backend function, e.g.
        #     keras.backend.split(a, 4, axis=1)
        a_x1, a_y1, b_x1, b_y1 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:]

        # TODO: Write `split` Keras backend function, e.g.
        #     keras.backend.split(b, 4, axis=1)
        a_x2, a_y2, b_x2, b_y2 = b[:, 0:1], b[:, 1:2], b[:, 2:3], b[:, 3:]

        x_intersection_1 = keras.backend.maximum(a_x1, keras.backend.transpose(a_x2))
        y_intersection_1 = keras.backend.maximum(a_y1, keras.backend.transpose(a_y2))

        x_intersection_2 = keras.backend.minimum(b_x1, keras.backend.transpose(b_x2))
        y_intersection_2 = keras.backend.minimum(b_y1, keras.backend.transpose(b_y2))

        a = keras.backend.maximum(x_intersection_2 - x_intersection_1 + 1, 0)
        b = keras.backend.maximum(y_intersection_2 - y_intersection_1 + 1, 0)

        intersection = a * b

        bounding_boxes_a_area = (b_x1 - a_x1 + 1) * (b_y1 - a_y1 + 1)
        bounding_boxes_b_area = (b_x2 - a_x2 + 1) * (b_y2 - a_y2 + 1)

        union = (bounding_boxes_a_area + keras.backend.transpose(bounding_boxes_b_area)) - intersection

        return keras.backend.maximum(intersection / union, 0)

    def categorical_crossentropy(self, target, output):
        output = keras.backend.minimum(output, keras.backend.epsilon())

        output = keras.backend.transpose(keras.backend.log(output))

        return -keras.backend.dot(target, output)

    def compute_mask_loss(
            self,
            target_bounding_boxes,
            output_bounding_boxes,
            target_masks,
            output_masks
    ):
        target_bounding_boxes = keras.backend.squeeze(target_bounding_boxes, axis=0)
        output_bounding_boxes = keras.backend.squeeze(output_bounding_boxes, axis=0)

        index = keras.backend.prod(keras.backend.shape(target_masks)[2:])

        target_masks = keras.backend.reshape(target_masks, [-1, index])
        output_masks = keras.backend.reshape(output_masks, [-1, index])

        score = self.intersection_over_union(
            target_bounding_boxes,
            output_bounding_boxes
        )

        a = self.categorical_crossentropy(target_masks, output_masks)

        b = keras.backend.greater(score, self.threshold)
        b = keras.backend.cast(b, dtype=keras.backend.floatx())

        return keras.backend.mean(a * b) + keras.backend.mean(1 - b)

    def compute_output_shape(self, input_shape):
        return input_shape[3]
