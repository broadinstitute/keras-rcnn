import keras


# def proposal(anchors, *args, **kwargs):
#     def f(y_true, y_pred):
#         # separate y_pred into rpn_cls_pred and rpn_reg_pred
#         y_pred_classification, y_pred_regression = separate_pred(y_pred)
#
#         # convert y_true from gt_boxes to gt_anchors
#         y_true_classification, y_true_regression = encode(y_true, y_pred_classification)
#
#         classification = _classification(anchors=anchors)(y_true_classification, y_pred_classification)
#
#         regression = _regression(anchors=anchors)(y_true_regression, y_pred_regression)
#
#         return classification + regression
#
#     return f


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
