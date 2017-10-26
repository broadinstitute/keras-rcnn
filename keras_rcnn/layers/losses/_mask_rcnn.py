# -*- coding: utf-8 -*-

import keras.backend
import keras.layers


class RCNNMaskLoss(keras.layers.Layer):
    def __init__(self, threshold=0.5, **kwargs):
        self.threshold = threshold

        super(RCNNMaskLoss, self).__init__(**kwargs)

    def call(self, inputs, training=None, **kwargs):
        target_boxes, predicred_boxes, target_masks, predicted_masks = inputs

        """
        loss = keras.backend.in_train_phase(
            lambda: self.mask_loss(target_boundingbox=target_boxes, 
                                   detected_boundingbox=predicred_boxes, 
                                   target_mask=target_masks, 
                                   detected_mask=predicted_masks,
                                   threshold=self.threshold),
            keras.backend.variable(0),
            training=training
        )
        """
        loss = self.compute_mask_loss(
            target_bounding_box=target_boxes,
            output_bounding_box=predicred_boxes,
            target_mask=target_masks,
            output_mask=predicted_masks,
            threshold=self.threshold
        )

        self.add_loss(loss, inputs)

        return predicted_masks

    @staticmethod
    def intersection_over_union(a, b):
        """
        Args:
            a: shape (total_bboxes1, 4)
                with x1, y1, x2, y2 point order.

            b: shape (total_bboxes2, 4)
                with x1, y1, x2, y2 point order.

            p1 *-----
               |     |
               |_____* p2

        Returns:
            Tensor with shape (total_bboxes1, total_bboxes2)
            with the IoU (intersection over union) of bboxes1[i] and bboxes2[j]
            in [i, j].
        """

        a_x1, a_y1, b_x1, b_y1 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:]  # tf.split(bboxes1, 4, axis=1)
        a_x2, a_y2, b_x2, b_y2 = b[:, 0:1], b[:, 1:2], b[:, 2:3], b[:, 3:]  # tf.split(bboxes2, 4, axis=1)

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

    @staticmethod
    def binary_crossentropy(_sentinel=None, target=None, output=None):
        """
        Args:
            _sentinel: internal use only
            labels: target image (n_masks2,N)
            output: network output after softmax or sigmoid of size (n_masks1,N)
        Returns:
            Tensor with shape (n_masks1, n_masks2)
            with the binary cross entropy between probs and labels
            in [i,j]
        """
        epsilon = keras.backend.epsilon()

        intermediate = keras.backend.dot(target, keras.backend.transpose(keras.backend.log(output + epsilon))) + keras.backend.dot((1. - target), keras.backend.transpose(keras.backend.log(1. - output + epsilon)))

        return - intermediate / keras.backend.cast(keras.backend.shape(target)[1], dtype=keras.backend.floatx())

    @staticmethod
    def categorical_crossentropy(_sentinel=None, target=None, output=None):
        """
        Args:
            _sentinel: internal use only
            output: network output after softmax or sigmoid of size (n_masks1,N)
            labels: target image (n_masks2,N)
        Returns:
            Tensor with shape (n_masks1, n_masks2)
            with the categorial cross entropy between probs and labels
            in [i,j]
        """
        epsilon = keras.backend.epsilon()

        # TODO: normalize dot product, size of the logits / number of classes
        return -keras.backend.dot(target, keras.backend.transpose(keras.backend.log(output + epsilon)))

    @staticmethod
    def compute_mask_loss(_sentinel=None, target_bounding_box=None, output_bounding_box=None, target_mask=None, output_mask=None, threshold=0.5):
        """
        Args:
            _sentinel: internal use only
            target_bounding_box: ground truth bounding boxes (1,total_boxes1,4)
            output_bounding_box: predicted bounding boxes (1,total_boxes2,4)
            target_mask: ground truth masks (1,total_masks1,N,M)
            output_mask: predicted masks (1,total_masks2,N,M)
            threshold: a scalar iou value after which bounding box is valid for bce
        Retruns:
            Mean binary cross entropy if IoU between bounding boxes is greater than threshold

        """
        target_bounding_box = keras.backend.squeeze(target_bounding_box, axis=0)
        output_bounding_box = keras.backend.squeeze(output_bounding_box, axis=0)

        index = keras.backend.prod(keras.backend.shape(target_mask)[2:])

        target_mask = keras.backend.reshape(target_mask, [-1, index])
        output_mask = keras.backend.reshape(output_mask, [-1, index])

        iou = RCNNMaskLoss.intersection_over_union(target_bounding_box, output_bounding_box)

        a = RCNNMaskLoss.categorical_crossentropy(target_mask, output_mask)

        b = keras.backend.greater(iou, threshold)
        b = keras.backend.cast(b, dtype=keras.backend.floatx())

        # TODO: we should try:
        #   `keras.backend.mean(a * b) + keras.backend.mean(1 - b)`
        return keras.backend.mean(a * b)

    def compute_output_shape(self, input_shape):
        return input_shape[3]
