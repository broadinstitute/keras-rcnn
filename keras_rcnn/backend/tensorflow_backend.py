import keras.backend
import tensorflow

import keras_rcnn.backend

RPN_NEGATIVE_OVERLAP = 0.3
RPN_POSITIVE_OVERLAP = 0.7
RPN_FG_FRACTION = 0.5
RPN_BATCHSIZE = 256


def scatter_add_tensor(ref, indices, updates, name=None):
    """
    Adds sparse updates to a variable reference.

    This operation outputs ref after the update is done. This makes it easier to chain operations that need to use the
    reset value.

    Duplicate indices are handled correctly: if multiple indices reference the same location, their contributions add.

    Requires updates.shape = indices.shape + ref.shape[1:].
    :param ref: A Tensor. Must be one of the following types: float32, float64, int64, int32, uint8, uint16,
        int16, int8, complex64, complex128, qint8, quint8, qint32, half.
    :param indices: A Tensor. Must be one of the following types: int32, int64. A tensor of indices into the first
        dimension of ref.
    :param updates: A Tensor. Must have the same dtype as ref. A tensor of updated values to add to ref
    :param name: A name for the operation (optional).
    :return: Same as ref. Returned as a convenience for operations that want to use the updated values after the update
        is done.
    """
    with tensorflow.name_scope(name, 'scatter_add_tensor', [ref, indices, updates]) as scope:
        ref = tensorflow.convert_to_tensor(ref, name='ref')
        indices = tensorflow.convert_to_tensor(indices, name='indices')
        updates = tensorflow.convert_to_tensor(updates, name='updates')
        ref_shape = tensorflow.shape(ref, out_type=indices.dtype, name='ref_shape')
        scattered_updates = tensorflow.scatter_nd(indices, updates, ref_shape, name='scattered_updates')
        with tensorflow.control_dependencies([tensorflow.assert_equal(ref_shape, tensorflow.shape(scattered_updates, out_type=indices.dtype))]):
            output = tensorflow.add(ref, scattered_updates, name=scope)
        return output


def bbox_transform_inv(shifted, boxes):
    def shape_zero():
        return keras.backend.zeros_like(keras.backend.shape(boxes)[1],
                                        dtype=boxes.dtype)

    def shape_non_zero():
        a = shifted[:, 2] - shifted[:, 0] + 1.0
        b = shifted[:, 3] - shifted[:, 1] + 1.0
        ctr_x = shifted[:, 0] + 0.5 * a
        ctr_y = shifted[:, 1] + 0.5 * b
        dx = boxes[:, 0::4]
        dy = boxes[:, 1::4]
        dw = boxes[:, 2::4]
        dh = boxes[:, 3::4]
        pred_ctr_x = dx * a[:, tensorflow.newaxis] + ctr_x[:,
                                                     tensorflow.newaxis]
        pred_ctr_y = dy * b[:, tensorflow.newaxis] + ctr_y[:,
                                                     tensorflow.newaxis]
        pred_w = keras.backend.exp(dw) * a[:, tensorflow.newaxis]
        pred_h = keras.backend.exp(dh) * b[:, tensorflow.newaxis]
        pred_boxes = [pred_ctr_x - 0.5 * pred_w, pred_ctr_y - 0.5 * pred_h,
                      pred_ctr_x + 0.5 * pred_w, pred_ctr_y + 0.5 * pred_h]
        return keras.backend.concatenate(pred_boxes)

    zero_boxes = tensorflow.equal(keras.backend.shape(boxes)[0], 0)
    pred_boxes = tensorflow.cond(zero_boxes, shape_zero, shape_non_zero)
    return pred_boxes


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
    shape = keras.backend.shape(boxes)[1:3]

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

    indicies = keras_rcnn.backend.non_maximum_suppression(proposals, scores,
                                                          maximum, 0.7)

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
    box_ind = tensorflow.reshape(box_ind, [-1])

    boxes = tensorflow.reshape(boxes, [-1, 4])
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
    area = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)
    iw = tensorflow.minimum(keras.backend.expand_dims(a[:, 2], 1),
                            b[:, 2]) - tensorflow.maximum(
        keras.backend.expand_dims(a[:, 0], 1), b[:, 0]) + 1
    ih = tensorflow.minimum(keras.backend.expand_dims(a[:, 3], 1),
                            b[:, 3]) - tensorflow.maximum(
        keras.backend.expand_dims(a[:, 1], 1), b[:, 1]) + 1
    iw = tensorflow.maximum(iw, 0)
    ih = tensorflow.maximum(ih, 0)
    ua = tensorflow.expand_dims(
        (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1), 1) + area - iw * ih
    ua = tensorflow.maximum(ua, 0.0001)

    return iw * ih / ua


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

    max_overlaps = tensorflow.gather_nd(overlaps, tensorflow.transpose(
        tensorflow.stack(
            [tensorflow.range(keras.backend.shape(inds_inside)[0]),
             keras.backend.cast(argmax_overlaps_inds, tensorflow.int32)],
            axis=0)))

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

    fg_inds = keras.backend.shape(tensorflow.where(tensorflow.equal(labels, 1)))[0]

    size = tensorflow.subtract(tensorflow.cast(fg_inds, tensorflow.int32), tensorflow.cast(num_fg, tensorflow.int32))

    def more_positive():
        elems = tensorflow.multinomial(tensorflow.log(tensorflow.ones((1, fg_inds))), size)
        indices = tensorflow.reshape(elems, (-1, 1))
        return scatter_add_tensor(labels, indices, tensorflow.ones((size,)) * -1)

    def less_positive():
        return labels

    return tensorflow.cond(tensorflow.less_equal(size, 0), lambda: less_positive(), lambda: more_positive())


def subsample_negative_labels(labels):
    """
    subsample negative labels if we have too many
    :param labels: array of labels (1 is positive, 0 is negative, -1 is dont care)

    :return:
    """
    num_bg = RPN_BATCHSIZE - keras.backend.shape(tensorflow.where(keras.backend.equal(labels, 1)))[0]

    bg_inds = keras.backend.shape(tensorflow.where(keras.backend.equal(labels, 0)))[0]

    size = bg_inds - num_bg

    def more_negative():
        elems = tensorflow.multinomial(keras.backend.log(tensorflow.ones((1, bg_inds))), size)

        ref = labels

        indices = keras.backend.reshape(elems, (-1, 1))

        updates = tensorflow.ones((size,)) * -1

        return scatter_add_tensor(ref, indices, updates)

    def less_negative():
        return labels

    predicate = keras.backend.less_equal(size, 0)

    return tensorflow.cond(predicate, lambda: less_negative(), lambda: more_negative())


def label(y_true, y_pred, inds_inside):
    """
    Create bbox labels.
    label: 1 is positive, 0 is negative, -1 is dont care

    :param inds_inside:
    :param y_pred: anchors
    :param y_true:

    :return:
    """
    labels = keras.backend.ones_like(inds_inside,
                                     dtype=tensorflow.float32) * -1
    argmax_overlaps_inds, max_overlaps, gt_argmax_overlaps_inds = overlapping(
        y_true, y_pred, inds_inside)

    # assign bg labels first so that positive labels can clobber them
    comparison = keras.backend.less(max_overlaps, keras.backend.constant(
        RPN_NEGATIVE_OVERLAP))
    labels = tensorflow.assign(
        tensorflow.Variable(labels, validate_shape=False),
        tensorflow.where(comparison, keras.backend.zeros_like(max_overlaps,
                                                              dtype=tensorflow.float32),
                         labels))

    # fg label: for each gt, anchor with highest overlap
    labels = tensorflow.scatter_nd_update(labels, tensorflow.expand_dims(
        gt_argmax_overlaps_inds, axis=1), keras.backend.ones_like(
        gt_argmax_overlaps_inds, dtype=tensorflow.float32))

    # fg label: above threshold IOU
    comparison = keras.backend.greater_equal(max_overlaps,
                                             keras.backend.constant(
                                                 RPN_POSITIVE_OVERLAP))
    labels = tensorflow.assign(labels, tensorflow.where(comparison,
                                                        keras.backend.ones_like(
                                                            max_overlaps,
                                                            dtype=tensorflow.float32),
                                                        labels))

    # assign bg labels last so that negative labels can clobber positives
    comparison = keras.backend.less(max_overlaps, keras.backend.constant(
        RPN_NEGATIVE_OVERLAP))
    labels = tensorflow.assign(labels, tensorflow.where(comparison,
                                                        keras.backend.zeros_like(
                                                            max_overlaps,
                                                            dtype=tensorflow.float32),
                                                        labels))

    return argmax_overlaps_inds, balance(labels)


def shift(shape, stride):
    shift_x = keras.backend.arange(0, shape[0]) * stride
    shift_y = keras.backend.arange(0, shape[1]) * stride

    shift_x, shift_y = tensorflow.meshgrid(shift_x, shift_y)

    shifts = tensorflow.stack((tensorflow.reshape(shift_x, [-1]),
                               tensorflow.reshape(shift_y, [-1]),
                               tensorflow.reshape(shift_x, [-1]),
                               tensorflow.reshape(shift_y, [-1])), axis=0)
    shifts = tensorflow.transpose(shifts)
    anchors = keras_rcnn.backend.anchor()

    # Create all bbox
    number_of_anchors = tensorflow.shape(anchors)[0]

    k = tensorflow.shape(shifts)[0]  # number of base points = feat_h * feat_w

    bbox = tensorflow.reshape(anchors,
                              [1, number_of_anchors, 4]) + tensorflow.cast(
        tensorflow.reshape(shifts, [k, 1, 4]), dtype=tensorflow.float32)

    bbox = tensorflow.reshape(bbox, [k * number_of_anchors, 4])

    return bbox


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
    inds_inside = keras.backend.cast(inds_inside, tensorflow.int32)

    return inds_inside[:, 0], tensorflow.reshape(
        tensorflow.gather(y_pred, inds_inside), [-1, 4])
