import keras.backend
import tensorflow

import keras_rcnn.backend

RPN_NEGATIVE_OVERLAP = 0.3
RPN_POSITIVE_OVERLAP = 0.7
RPN_FG_FRACTION = 0.5
RPN_BATCHSIZE = 256


def bbox_transform_inv(shifted, boxes):
    if boxes.shape[0] == 0:
        return keras.backend.zeros((0, boxes.shape[1]), dtype=boxes.dtype)

    a = shifted[:, 2] - shifted[:, 0] + 1.0
    b = shifted[:, 3] - shifted[:, 1] + 1.0

    ctr_x = shifted[:, 0] + 0.5 * a
    ctr_y = shifted[:, 1] + 0.5 * b

    dx = boxes[:, 0::4]
    dy = boxes[:, 1::4]
    dw = boxes[:, 2::4]
    dh = boxes[:, 3::4]

    pred_ctr_x = dx * a[:, tensorflow.newaxis] + ctr_x[:, tensorflow.newaxis]
    pred_ctr_y = dy * b[:, tensorflow.newaxis] + ctr_y[:, tensorflow.newaxis]

    pred_w = keras.backend.exp(dw) * a[:, tensorflow.newaxis]
    pred_h = keras.backend.exp(dh) * b[:, tensorflow.newaxis]

    pred_boxes = [
        pred_ctr_x - 0.5 * pred_w,
        pred_ctr_y - 0.5 * pred_h,
        pred_ctr_x + 0.5 * pred_w,
        pred_ctr_y + 0.5 * pred_h
    ]

    return keras.backend.concatenate(pred_boxes)


def filter_boxes(proposals, minimum):
    ws = proposals[:, 2] - proposals[:, 0] + 1
    hs = proposals[:, 3] - proposals[:, 1] + 1

    indicies = tensorflow.where((ws >= minimum) & (hs >= minimum))

    indicies = keras.backend.flatten(indicies)

    return keras.backend.cast(indicies, tensorflow.int32)


def non_maximum_suppression(boxes, scores, maximum, threshold=0.5):
    return tensorflow.image.non_max_suppression(
        boxes=boxes,
        iou_threshold=threshold,
        max_output_size=maximum,
        scores=scores
    )


def propose(boxes, scores, maximum):
    shape = keras.backend.int_shape(boxes)[1:3]

    shifted = keras_rcnn.backend.shift(shape, 16)

    proposals = keras.backend.reshape(boxes, (-1, 4))

    proposals = keras_rcnn.backend.bbox_transform_inv(shifted, proposals)

    proposals = keras_rcnn.backend.clip(proposals, shape)

    indicies = keras_rcnn.backend.filter_boxes(proposals, 1)

    proposals = keras.backend.gather(proposals, indicies)

    scores = scores[:, :, :, :9]
    scores = keras.backend.reshape(scores, (-1, 1))
    scores = keras.backend.gather(scores, indicies)
    scores = keras.backend.flatten(scores)

    proposals = keras.backend.cast(proposals, tensorflow.float32)

    scores = keras.backend.cast(scores, tensorflow.float32)

    indicies = keras_rcnn.backend.non_maximum_suppression(proposals, scores, maximum, 0.7)

    proposals = keras.backend.gather(proposals, indicies)

    return keras.backend.expand_dims(proposals, 0)


def resize_images(images, shape):
    return tensorflow.image.resize_images(images, shape)


def crop_and_resize(image, boxes, size):
    """Crop the image given boxes and resize with bilinear interplotation.
    # Parameters
    image: Input image of shape (1, image_height, image_width, depth)
    boxes: Regions of interest of shape (1, num_boxes, 4),
    each row [y1, x1, y2, x2]
    size: Fixed size [h, w], e.g. [7, 7], for the output slices.
    # Returns
    4D Tensor (number of regions, slice_height, slice_width, channels)
    """
    box_ind = keras.backend.zeros_like(boxes, tensorflow.int32)
    box_ind = box_ind[..., 0]
    box_ind = keras.backend.reshape(box_ind, [-1])

    boxes = keras.backend.reshape(boxes, [-1, 4])

    return tensorflow.image.crop_and_resize(image, boxes, box_ind, size)


def overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    n = tensorflow.shape(a)[0]

    k = tensorflow.shape(b)[0]

    i = tensorflow.constant(0)

    initial_overlaps = tensorflow.Variable([])

    def cond(i, l):
        return i < n

    def body(i, l):
        area = ((b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1))

        iw = tensorflow.maximum( (tensorflow.minimum(a[i, 2], b[:, 2]) - tensorflow.maximum(a[i, 0], b[:, 0]) + 1), 0.0 )

        ih = tensorflow.maximum( (tensorflow.minimum(a[i, 3], b[:, 3]) - tensorflow.maximum(a[i, 1], b[:, 1]) + 1), 0.0 )

        ua = (a[i, 2] - a[i, 0] + 1) * (a[i, 3] - a[i, 1] + 1) + area - iw * ih

        iou = iw * ih / ua

        l = tensorflow.concat([l, iou], 0)

        return i+1, l

    index, final_overlaps = tensorflow.while_loop(
        cond,
        body,
        [i, initial_overlaps],
        shape_invariants=[i.get_shape(), tensorflow.TensorShape([None])]
    )

    final_overlaps = tensorflow.reshape(final_overlaps, (n, k))

    return final_overlaps


def overlapping(y_true, y_pred, inds_inside):
    """
    overlaps between the anchors and the gt boxes
    :param y_pred: anchors
    :param y_true:
    :param inds_inside:
    :return:
    """
    overlaps = overlap(y_pred, y_true[:, :4])

    argmax_overlaps_inds = tensorflow.argmax(overlaps, axis=1)

    gt_argmax_overlaps_inds = tensorflow.argmax(overlaps, axis=0)

    max_overlaps = tensorflow.gather_nd(overlaps, tensorflow.transpose(tensorflow.stack([tensorflow.range(tensorflow.shape(inds_inside)[0]), tensorflow.cast(argmax_overlaps_inds, tensorflow.int32)], axis=0)))

    return argmax_overlaps_inds, max_overlaps, gt_argmax_overlaps_inds


def balance(labels):
    """
    balance labels by setting some to -1
    :param labels: array of labels (1 is positive, 0 is negative, -1 is dont care)
    :return: array of labels
    """
    # subsample positive labels if we have too many
    labels = subsample_positive_labels(labels)

    # subsample negative labels if we have too many
    labels = subsample_negative_labels(labels)

    return labels


def subsample_positive_labels(labels):
    """
    subsample positive labels if we have too many
    :param labels: array of labels (1 is positive, 0 is negative, -1 is dont care)
    :return:
    """
    num_fg = RPN_FG_FRACTION * RPN_BATCHSIZE

    fg_inds = tensorflow.where(tensorflow.equal(labels, 1))

    fg_inds = tensorflow.shape(fg_inds)[0]

    size = tensorflow.cast(fg_inds, tensorflow.int32) - tensorflow.cast(num_fg, tensorflow.int32)

    def more_positive():
        print(tensorflow.multinomial(tensorflow.log(tensorflow.ones((fg_inds, 1)) * 10.), size))

        elems = tensorflow.gather(tensorflow.range(fg_inds), tensorflow.multinomial(tensorflow.log(tensorflow.ones((fg_inds, 1)) * 10.), size))
        
        return tensorflow.scatter_update(tensorflow.Variable(labels, validate_shape=False), elems, -1)

    def less_positive():
        return labels

    return tensorflow.cond(tensorflow.less_equal(size, 0), lambda: less_positive(), lambda: more_positive())


def subsample_negative_labels(labels):
    """
    subsample negative labels if we have too many
    :param labels: array of labels (1 is positive, 0 is negative, -1 is dont care)
    :return:
    """
    num_bg = RPN_BATCHSIZE - tensorflow.reduce_sum(tensorflow.gather(labels, tensorflow.where(tensorflow.equal(labels, 1))))
    
    bg_inds = tensorflow.where(tensorflow.equal(labels, 0))
    
    bg_inds = tensorflow.shape(bg_inds)[0]
    
    size = tensorflow.cast(bg_inds, tensorflow.int32) - tensorflow.cast(num_bg, tensorflow.int32)
    
    def more_negative():
        elems = tensorflow.gather(tensorflow.range(bg_inds), tensorflow.multinomial(tensorflow.log(tensorflow.ones((bg_inds, 1)) * 10.), size))
        
        return tensorflow.scatter_update(tensorflow.Variable(labels, validate_shape=False), elems, -1)

    def less_negative():
        return labels

    return tensorflow.cond(tensorflow.less_equal(size, 0), lambda: less_negative(), lambda: more_negative())


def shift(shape, stride):
    shift_r = keras.backend.arange(0, shape[0]) * stride
    shift_c = keras.backend.arange(0, shape[1]) * stride

    shift_r, shift_c = tensorflow.meshgrid(shift_r, shift_c)

    shifted_anchors = keras.backend.stack(
        (
            keras.backend.reshape(shift_r, [-1]),
            keras.backend.reshape(shift_c, [-1]),
            keras.backend.reshape(shift_r, [-1]),
            keras.backend.reshape(shift_c, [-1])
        )
    )

    shifted_anchors = keras.backend.transpose(shifted_anchors)

    anchors = keras_rcnn.backend.anchor()

    number_of_anchors = keras.backend.shape(anchors)[0]

    # number of base points, k = feat_h * feat_w
    k = keras.backend.shape(shifted_anchors)[0]

    boxes = keras.backend.reshape(anchors, [1, number_of_anchors, 4])
    boxes = keras.backend.cast(boxes, keras.backend.floatx())

    shifted_anchors = keras.backend.reshape(shifted_anchors, [k, 1, 4])
    shifted_anchors = keras.backend.cast(shifted_anchors, keras.backend.floatx())

    boxes += shifted_anchors

    boxes = keras.backend.reshape(boxes, [k * number_of_anchors, 4])

    return boxes


def inside_image(y_pred, img_info):
    """
    Calc indicies of anchors which are located completely inside of the image
    whose size is specified by img_info ((height, width, scale)-shaped array).
    :param y_pred: anchors
    :param img_info:
    :return:
    """
    inds_inside = tensorflow.where(
        (y_pred[:, 0] >= 0) &
        (y_pred[:, 1] >= 0) &
        (y_pred[:, 2] < img_info[1]) &  # width
        (y_pred[:, 3] < img_info[0])  # height
    )

    inds_inside = tensorflow.cast(inds_inside, tensorflow.int32)

    return inds_inside[:, 0], keras.backend.reshape(keras.backend.gather(y_pred, inds_inside), [keras.backend.shape(inds_inside)[0], 4])
