import keras.backend
import keras_rcnn.backend
import numpy
import tensorflow
import keras.losses


def separate_pred(y_pred):
    anchors = tensorflow.shape(y_pred)[-1] // 5
    return y_pred[:, :, :, 0: 4 * anchors], y_pred[:, :, :,
                                            4 * anchors: 5 * anchors]


def encode(features, image_shape, y_true, stride=16):
    # get all anchors inside bbox
    shifted_anchors = keras_rcnn.backend.shift(features, stride)
    inds_inside, all_inside_bbox = keras_rcnn.backend.inside_image(
        shifted_anchors, image_shape)

    # indices of gt boxes with the greatest overlap, bbox labels
    # TODO: assert y_true.shape[0] == 1
    y_true = y_true[0, :, :]
    argmax_overlaps_inds, bbox_labels = keras_rcnn.backend.label(y_true,
                                                                 all_inside_bbox,
                                                                 inds_inside)

    # gt boxes
    gt_boxes = tensorflow.gather(y_true, argmax_overlaps_inds)

    # Convert fixed anchors in (x, y, w, h) to (dx, dy, dw, dh)
    bbox_reg_targets = keras_rcnn.backend.bbox_transform(all_inside_bbox,
                                                         gt_boxes)

    return bbox_labels, bbox_reg_targets, inds_inside


def proposal(anchors, image_shape, stride, *args, **kwargs):
    def f(y_true, y_pred):
        # separate y_pred into rpn_cls_pred and rpn_reg_pred
        y_pred_regression, y_pred_classification = separate_pred(y_pred)

        # convert y_true from gt_boxes to gt_anchors
        features = y_pred.get_shape().as_list()[1:3]

        gt_classification, gt_regression, inds_inside = encode(features,
                                                               image_shape,
                                                               y_true, stride)
        y_true_classification = tensorflow.zeros(
            (1, features[0], features[1], anchors * 2))
        y_true_regression = tensorflow.zeros(
            (1, features[0], features[1], anchors * 8))

        z = tensorflow.expand_dims(tensorflow.zeros_like(inds_inside), 1)
        i = tensorflow.expand_dims(inds_inside // (anchors * features[1]), 1)
        j = tensorflow.expand_dims(inds_inside // anchors % features[1], 1)
        a = tensorflow.expand_dims(
            (inds_inside % anchors) * tensorflow.cast(gt_classification,
                                                      tensorflow.int32), 1)
        zz = tensorflow.reshape(tensorflow.tile(z, [1, 4]), (-1, 1))
        ii = tensorflow.reshape(tensorflow.tile(i, [1, 4]), (-1, 1))
        jj = tensorflow.reshape(tensorflow.tile(j, [1, 4]), (-1, 1))
        aa = tensorflow.concat([a, a + 1, a + 2, a + 3], 0)
        gt_r1 = tensorflow.tile(gt_classification, [4])

        y_true_classification = tensorflow.scatter_nd_update(
            tensorflow.Variable(y_true_classification, validate_shape=False),
            tensorflow.concat([z, i, j, a], 1), gt_classification)
        y_true_classification = tensorflow.scatter_nd_update(
            y_true_classification,
            tensorflow.concat([z, i, j, anchors + a], 1), gt_classification)
        y_true_regression = tensorflow.scatter_nd_update(
            tensorflow.Variable(y_true_regression, validate_shape=False),
            tensorflow.concat([zz, ii, jj, aa], 1), gt_r1)
        y_true_regression = tensorflow.scatter_nd_update(y_true_regression,
                                                         tensorflow.concat(
                                                             [zz, ii, jj,
                                                              anchors * 4 + aa],
                                                             1),
                                                         tensorflow.reshape(
                                                             gt_regression,
                                                             (-1,)))

        y_true_classification = tensorflow.reshape(y_true_classification, (
        1, features[0], features[1], anchors * 2))
        y_true_regression = tensorflow.reshape(y_true_regression, (
        1, features[0], features[1], anchors * 8))

        classification = _classification(anchors=anchors)(
            y_true_classification, y_pred_classification)

        regression = _regression(anchors=anchors)(y_true_regression,
                                                  y_pred_regression)
        loss = classification + regression

        return loss

    return f


def _classification(anchors=9):
    """
    Return the classification loss of region proposal network.

    :param anchors: Integer, number of anchors at each sliding position. Equal to number of scales * number of aspect ratios.

    :return: A loss function for region propose classification.
    """

    def f(y_true, y_pred):
        # Binary classification loss
        x, y = y_pred[:, :, :, :], y_true[:, :, :, anchors:]

        a = y_true[:, :, :, :anchors] * keras.backend.binary_crossentropy(x, y)
        a = keras.backend.sum(a)

        # Divided by anchor overlaps
        b = keras.backend.epsilon() + y_true[:, :, :, :anchors]
        b = keras.backend.sum(b)

        return 1.0 * (a / b)

    return f


def _regression(anchors=9):
    """
    Return the regression loss of region proposal network.

    :param anchors: Integer, number of anchors at each sliding position. Equal to number of scales * number of aspect ratios.

    :return: A loss function region propose regression.
    """

    def f(y_true, y_pred):
        # Robust L1 Loss
        x = y_true[:, :, :, 4 * anchors:] - y_pred

        mask = keras.backend.less_equal(keras.backend.abs(x), 1.0)
        mask = keras.backend.cast(mask, keras.backend.floatx())

        a_x = y_true[:, :, :, :4 * anchors]

        a_y = mask * (0.5 * x * x) + (1 - mask) * (keras.backend.abs(x) - 0.5)

        a = a_x * a_y
        a = keras.backend.sum(a)

        # Divided by anchor overlaps
        b = keras.backend.epsilon() + a_x
        b = keras.backend.sum(b)

        return 1.0 * (a / b)

    return f
