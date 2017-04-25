"""

For training a region proposal network (RPN), we assign a binary class label
(of being an object or not) to each anchor. We assign a positive label to two
kinds of anchors:

    (i) the anchor/anchors with the highest Intersection-over-Union (IoU)
    overlap with a ground-truth box, or

    (ii) an anchor that has an IoU overlap higher than 0.7 with any
    ground-truth box.

A single ground-truth box may assign positive labels to multiple anchors.
Usually the second condition is sufficient to determine the positive samples;
but we still adopt the first condition for the reason that in some rare cases
the second condition may find no positive sample. We assign a negative label to
a non-positive anchor if its IoU ratio is lower than 0.3 for all ground-truth
boxes. Anchors that are neither positive nor negative do not contribute to the
training objective.

With these definitions, we minimize an objective function following the
multi-task loss in Fast R-CNN [2].

Our loss function for an image is defined as:

"""

import keras
import keras.backend
import keras.objectives


def classification(anchors):
    def f(y_true, y_pred):
        x, y = y_pred[:, :, :, :], y_true[:, :, :, anchors:]

        a = y_true[:, :, :, :anchors] * keras.backend.binary_crossentropy(x, y)
        a = keras.backend.sum(a)

        b = keras.backend.epsilon() + y_true[:, :, :, :anchors]
        b = keras.backend.sum(b)

        return 1.0 * (a / b)

    return f


def regression(anchors):
    def f(y_true, y_pred):
        x = y_true[:, :, :, 4 * anchors:] - y_pred

        mask = keras.backend.less_equal(keras.backend.abs(x), 1.0)
        mask = keras.backend.cast(mask, keras.backend.floatx())

        a_x = y_true[:, :, :, :4 * anchors]

        a_y = mask * (0.5 * x * x) + (1 - mask) * (keras.backend.abs(x) - 0.5)

        a = a_x * a_y
        a = keras.backend.sum(a)

        b = keras.backend.epsilon() + a_x
        b = keras.backend.sum(b)

        return 1.0 * (a / b)

    return f
