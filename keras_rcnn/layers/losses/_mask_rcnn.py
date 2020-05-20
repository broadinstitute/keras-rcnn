# -*- coding: utf-8 -*-

import tensorflow


class RCNNMaskLoss(tensorflow.keras.layers.Layer):
    def __init__(self, threshold=0.5, **kwargs):
        self.threshold = threshold

        super(RCNNMaskLoss, self).__init__(**kwargs)

    def call(self, inputs, training=None, **kwargs):
        target_boxes, predicred_boxes, target_masks, predicted_masks = inputs

        """
        loss = tensorflow.keras.backend.in_train_phase(
            lambda: self.mask_loss(target_boundingbox=target_boxes, 
                                   detected_boundingbox=predicred_boxes, 
                                   target_mask=target_masks, 
                                   detected_mask=predicted_masks,
                                   threshold=self.threshold),
            tensorflow.keras.backend.variable(0),
            training=training
        )
        """

        loss = self.compute_mask_loss(
            target_bounding_box=target_boxes,
            output_bounding_box=predicred_boxes,
            target_mask=target_masks,
            output_mask=predicted_masks,
            threshold=self.threshold,
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

        a_x1, a_y1, b_x1, b_y1 = (
            a[:, 0:1],
            a[:, 1:2],
            a[:, 2:3],
            a[:, 3:],
        )  # tf.split(bboxes1, 4, axis=1)
        a_x2, a_y2, b_x2, b_y2 = (
            b[:, 0:1],
            b[:, 1:2],
            b[:, 2:3],
            b[:, 3:],
        )  # tf.split(bboxes2, 4, axis=1)

        x_intersection_1 = tensorflow.keras.backend.maximum(
            a_x1, tensorflow.keras.backend.transpose(a_x2)
        )
        y_intersection_1 = tensorflow.keras.backend.maximum(
            a_y1, tensorflow.keras.backend.transpose(a_y2)
        )

        x_intersection_2 = tensorflow.keras.backend.minimum(
            b_x1, tensorflow.keras.backend.transpose(b_x2)
        )
        y_intersection_2 = tensorflow.keras.backend.minimum(
            b_y1, tensorflow.keras.backend.transpose(b_y2)
        )

        a = tensorflow.keras.backend.maximum(x_intersection_2 - x_intersection_1 + 1, 0)
        b = tensorflow.keras.backend.maximum(y_intersection_2 - y_intersection_1 + 1, 0)

        intersection = a * b

        bounding_boxes_a_area = (b_x1 - a_x1 + 1) * (b_y1 - a_y1 + 1)
        bounding_boxes_b_area = (b_x2 - a_x2 + 1) * (b_y2 - a_y2 + 1)

        union = (
            bounding_boxes_a_area
            + tensorflow.keras.backend.transpose(bounding_boxes_b_area)
        ) - intersection

        return tensorflow.keras.backend.maximum(intersection / union, 0)

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
        epsilon = tensorflow.keras.backend.epsilon()

        intermediate = tensorflow.keras.backend.dot(
            target,
            tensorflow.keras.backend.transpose(
                tensorflow.keras.backend.log(output + epsilon)
            ),
        ) + tensorflow.keras.backend.dot(
            (1.0 - target),
            tensorflow.keras.backend.transpose(
                tensorflow.keras.backend.log(1.0 - output + epsilon)
            ),
        )

        return -intermediate / tensorflow.keras.backend.cast(
            tensorflow.keras.backend.shape(target)[1],
            dtype=tensorflow.keras.backend.floatx(),
        )

    @staticmethod
    def categorical_crossentropy(_sentinel=None, target=None, output=None):
        """
        Args:
            _sentinel: internal use only
            labels: target image (n_masks1,N)
            output: network output after softmax or sigmoid of size (n_masks2,N)
        Returns:
            Tensor with shape (n_masks1, n_masks2)
            with the categorical cross entropy between probs and labels
            in [i,j]
        """
        epsilon = tensorflow.keras.backend.epsilon()

        # TODO: normalize dot product, size of the logits / number of classes
        cce = -tensorflow.keras.backend.dot(
            target,
            tensorflow.keras.backend.transpose(
                tensorflow.keras.backend.log(output + epsilon)
            ),
        )
        return cce

    @staticmethod
    def compute_mask_loss(
        _sentinel=None,
        target_bounding_box=None,
        output_bounding_box=None,
        target_mask=None,
        output_mask=None,
        threshold=0.5,
    ):
        """
        Args:
            _sentinel: internal use only
            target_bounding_box: ground truth bounding boxes (1,total_boxes1,4)
            output_bounding_box: predicted bounding boxes (1,total_boxes2,4)
            target_mask: ground truth masks (1,total_boxes1,N,M)
            output_mask: predicted masks (1,total_boxes2,N,M)
            threshold: a scalar iou value after which bounding box is valid for bce
        Returns:
            Mean binary cross entropy if IoU between bounding boxes is greater than threshold

        """

        target_bounding_box = tensorflow.keras.backend.squeeze(
            target_bounding_box, axis=0
        )
        output_bounding_box = tensorflow.keras.backend.squeeze(
            output_bounding_box, axis=0
        )
        target_mask = tensorflow.keras.backend.squeeze(target_mask, axis=0)
        output_mask = tensorflow.keras.backend.squeeze(output_mask, axis=0)

        index = tensorflow.keras.backend.prod(
            tensorflow.keras.backend.shape(target_mask)[1:]
        )

        target_mask = tensorflow.keras.backend.reshape(target_mask, [-1, index])
        output_mask = tensorflow.keras.backend.reshape(output_mask, [-1, index])

        iou = RCNNMaskLoss.intersection_over_union(
            target_bounding_box, output_bounding_box
        )
        b = tensorflow.keras.backend.greater(iou, threshold)
        b = tensorflow.keras.backend.cast(b, dtype=tensorflow.keras.backend.floatx())

        # labels = tensorflow.keras.backend.greater(output_mask, 0.5)
        # labels = tensorflow.keras.backend.cast(labels, dtype=tensorflow.keras.backend.floatx())

        # labels = tensorflow.keras.backend.equal(labels, target_mask)
        # labels = tensorflow.keras.backend.cast(labels, dtype=tensorflow.keras.backend.floatx())

        a = RCNNMaskLoss.binary_crossentropy(target=target_mask, output=output_mask)

        # loss = tensorflow.keras.backend.sum(a * b) / tensorflow.keras.backend.sum(b)
        loss = tensorflow.keras.backend.mean(a * b)

        # TODO: we should try:
        #   `tensorflow.keras.backend.mean(a * b) + tensorflow.keras.backend.mean(1 - b)`

        return loss

    def compute_output_shape(self, input_shape):
        return input_shape[3]
