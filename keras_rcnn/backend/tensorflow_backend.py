import keras.backend
import tensorflow
import keras_rcnn.backend

RPN_FG_FRACTION = 0.5
RPN_BATCHSIZE = 256


def shuffle(x):
    """
    Modify a sequence by shuffling its contents. This function only shuffles
    the array along the first axis of a multi-dimensional array. The order of
    sub-arrays is changed but their contents remains the same.
    """
    return tensorflow.random_shuffle(x)


def gather_nd(params, indices):
    return tensorflow.gather_nd(params, indices)


def matmul(a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False, a_is_sparse=False, b_is_sparse=False):
    return tensorflow.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b, adjoint_a=adjoint_a, adjoint_b=adjoint_b, a_is_sparse=a_is_sparse, b_is_sparse=b_is_sparse)


# TODO: emulate NumPy semantics
def argsort(a):
    _, indices = tensorflow.nn.top_k(a, keras.backend.shape(a)[-1])

    return indices


def scatter_add_tensor(ref, indices, updates, name=None):
    """
    Adds sparse updates to a variable reference.

    This operation outputs ref after the update is done. This makes it easier to chain operations that need to use the
    reset value.

    Duplicate indices: if multiple indices reference the same location, their contributions add.

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


def meshgrid(*args, **kwargs):
    return tensorflow.meshgrid(*args, **kwargs)


newaxis = tensorflow.newaxis


def where(condition, x=None, y=None):
    return tensorflow.where(condition, x, y)


def bbox_transform_inv(shifted, boxes):
    def shape_zero():
        x = keras.backend.int_shape(boxes)[-1]

        return keras.backend.zeros_like(x, dtype=keras.backend.floatx())

    def shape_non_zero():
        a = shifted[:, 2] - shifted[:, 0] + 1.0
        b = shifted[:, 3] - shifted[:, 1] + 1.0

        ctr_x = shifted[:, 0] + 0.5 * a
        ctr_y = shifted[:, 1] + 0.5 * b

        dx = boxes[:, 0::4]
        dy = boxes[:, 1::4]
        dw = boxes[:, 2::4]
        dh = boxes[:, 3::4]

        pred_ctr_x = dx * a[:, keras_rcnn.backend.newaxis] + ctr_x[:, keras_rcnn.backend.newaxis]
        pred_ctr_y = dy * b[:, keras_rcnn.backend.newaxis] + ctr_y[:, keras_rcnn.backend.newaxis]

        pred_w = keras.backend.exp(dw) * a[:, keras_rcnn.backend.newaxis]
        pred_h = keras.backend.exp(dh) * b[:, keras_rcnn.backend.newaxis]

        prediction = [
            pred_ctr_x - 0.5 * pred_w, pred_ctr_y - 0.5 * pred_h,
            pred_ctr_x + 0.5 * pred_w, pred_ctr_y + 0.5 * pred_h
        ]

        return keras.backend.concatenate(prediction)

    zero_boxes = keras.backend.equal(keras.backend.shape(boxes)[0], 0)

    pred_boxes = tensorflow.cond(zero_boxes, shape_zero, shape_non_zero)

    return pred_boxes


def non_maximum_suppression(boxes, scores, maximum, threshold=0.5):
    return tensorflow.image.non_max_suppression(boxes=boxes, iou_threshold=threshold, max_output_size=maximum, scores=scores)


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
    box_ind = keras.backend.zeros_like(boxes, "int32")
    box_ind = box_ind[..., 0]
    box_ind = keras.backend.reshape(box_ind, [-1])

    boxes = keras.backend.reshape(boxes, [-1, 4])

    return tensorflow.image.crop_and_resize(image, boxes, box_ind, size)


def overlapping(anchors, gt_boxes, inds_inside):
    """
    overlaps between the anchors and the gt boxes
    :param anchors: Generated anchors
    :param gt_boxes: Ground truth bounding boxes
    :param inds_inside:
    :return:
    """

    assert keras.backend.ndim(anchors) == 2
    assert keras.backend.ndim(gt_boxes) == 2

    reference = keras_rcnn.backend.overlap(anchors, gt_boxes)

    gt_argmax_overlaps_inds = keras.backend.argmax(reference, axis=0)

    argmax_overlaps_inds = keras.backend.argmax(reference, axis=1)

    indices = keras.backend.stack([tensorflow.range(keras.backend.shape(inds_inside)[0]), keras.backend.cast(argmax_overlaps_inds, "int32")], axis=0)

    indices = keras.backend.transpose(indices)

    max_overlaps = tensorflow.gather_nd(reference, indices)

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

    num_fg = int(RPN_FG_FRACTION * RPN_BATCHSIZE)

    fg_inds = keras_rcnn.backend.where(keras.backend.equal(labels, 1))
    num_fg_inds = keras.backend.shape(fg_inds)[0]

    size = num_fg_inds - num_fg

    def more_positive():
        # TODO: try to replace tensorflow
        indices = tensorflow.random_shuffle(keras.backend.reshape(fg_inds, (-1,)))[:size]

        updates = tensorflow.ones((size,)) * -1

        inverse_labels = keras.backend.gather(labels, indices) * -1

        indices = keras.backend.reshape(indices, (-1, 1))

        return scatter_add_tensor(labels, indices, inverse_labels + updates)

    def less_positive():
        return labels

    predicate = keras.backend.less_equal(size, 0)

    return tensorflow.cond(predicate, lambda: less_positive(), lambda: more_positive())


def subsample_negative_labels(labels):
    """
    subsample negative labels if we have too many
    :param labels: array of labels (1 is positive, 0 is negative, -1 is dont care)

    :return:
    """
    num_bg = RPN_BATCHSIZE - keras.backend.shape(keras_rcnn.backend.where(keras.backend.equal(labels, 1)))[0]

    bg_inds = keras_rcnn.backend.where(keras.backend.equal(labels, 0))

    num_bg_inds = keras.backend.shape(bg_inds)[0]

    size = num_bg_inds - num_bg

    def more_negative():
        indices = keras_rcnn.backend.shuffle(keras.backend.reshape(bg_inds, (-1,)))[:size]

        updates = tensorflow.ones((size,)) * -1

        inverse_labels = keras.backend.gather(labels, indices) * -1

        indices = keras.backend.reshape(indices, (-1, 1))

        return scatter_add_tensor(labels, indices, inverse_labels + updates)

    def less_negative():
        return labels

    predicate = keras.backend.less_equal(size, 0)

    return tensorflow.cond(predicate, lambda: less_negative(), lambda: more_negative())


def label(y_true, y_pred, inds_inside, RPN_NEGATIVE_OVERLAP=0.3, RPN_POSITIVE_OVERLAP=0.7, clobber_positives=False):
    """
    Create bbox labels.
    label: 1 is positive, 0 is negative, -1 is do not care

    :param inds_inside: indices of anchors inside image
    :param y_pred: anchors
    :param y_true: ground truth objects

    :return: indices of gt boxes with the greatest overlap, balanced labels
    """
    ones = keras.backend.ones_like(inds_inside, dtype=keras.backend.floatx())
    labels = ones * -1
    zeros = keras.backend.zeros_like(inds_inside, dtype=keras.backend.floatx())

    argmax_overlaps_inds, max_overlaps, gt_argmax_overlaps_inds = overlapping(y_pred, y_true, inds_inside)

    # assign bg labels first so that positive labels can clobber them
    if not clobber_positives:
        labels = keras_rcnn.backend.where(keras.backend.less(max_overlaps, RPN_NEGATIVE_OVERLAP), zeros, labels)

    # fg label: for each gt, anchor with highest overlap

    # TODO: generalize unique beyond 1D
    unique_indices, unique_indices_indices = tensorflow.unique(gt_argmax_overlaps_inds, out_idx='int32')
    inverse_labels = keras.backend.gather(-1 * labels, unique_indices)
    unique_indices = keras.backend.expand_dims(unique_indices, 1)
    updates = keras.backend.ones_like(keras.backend.reshape(unique_indices, (-1,)), dtype=keras.backend.floatx())
    labels = keras_rcnn.backend.scatter_add_tensor(labels, unique_indices, inverse_labels + updates)

    # fg label: above threshold IOU
    labels = keras_rcnn.backend.where(keras.backend.greater_equal(max_overlaps, RPN_POSITIVE_OVERLAP), ones, labels)

    if clobber_positives:
        # assign bg labels last so that negative labels can clobber positives
        labels = keras_rcnn.backend.where(keras.backend.less(max_overlaps, RPN_NEGATIVE_OVERLAP), zeros, labels)

    return argmax_overlaps_inds, balance(labels)


def unmap(data, count, inds_inside, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """

    if keras.backend.ndim(data) == 1:
        ret = keras.backend.ones((count,), dtype=keras.backend.floatx()) * fill

        inds_nd = keras.backend.expand_dims(inds_inside)
    else:
        ret = keras.backend.ones((count,) + keras.backend.int_shape(data)[1:], dtype=keras.backend.floatx()) * fill
        data = keras.backend.transpose(data)
        data = keras.backend.reshape(data, (-1,))

        inds_ii = keras.backend.tile(inds_inside, [4])
        inds_ii = keras.backend.expand_dims(inds_ii)

        ones = keras.backend.expand_dims(keras.backend.ones_like(inds_inside), 1)

        inds_coords = keras.backend.concatenate([ones * 0, ones, ones * 2, ones * 3], 0)

        inds_nd = keras.backend.concatenate([inds_ii, inds_coords], 1)

    inverse_ret = tensorflow.squeeze(tensorflow.gather_nd(-1 * ret, inds_nd))

    ret = keras_rcnn.backend.scatter_add_tensor(ret, inds_nd, inverse_ret + data)

    return ret


def get_bbox_regression_labels(labels, bbox_target_data):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    form N x (tx, ty, tw, th), labels N x num_classes
    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).
    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
    """
    num_classes = keras.backend.shape(labels)[-1]

    clss = keras.backend.reshape(keras.backend.argmax(labels, axis=-1), (-1,))

    N = keras.backend.shape(bbox_target_data)[0]

    bbox_targets = tensorflow.zeros((N, 4 * num_classes), dtype=keras.backend.floatx())

    inds = keras.backend.reshape(keras_rcnn.backend.where(clss > 0), (-1,))

    cls = keras.backend.gather(clss, inds)

    start = 4 * cls

    ii = keras.backend.expand_dims(inds)
    ii = keras.backend.tile(ii, [4, 1])

    aa = keras.backend.expand_dims(keras.backend.concatenate([start, start + 1, start + 2, start + 3], 0))
    aa = keras.backend.cast(aa, dtype='int64')

    indices = keras.backend.concatenate([ii, aa], 1)

    updates = keras.backend.gather(bbox_target_data, inds)
    updates = keras.backend.transpose(updates)
    updates = keras.backend.reshape(updates, (-1,))

    # bbox_targets are 0
    bbox_targets = keras_rcnn.backend.scatter_add_tensor(bbox_targets, indices, updates)

    return bbox_targets


def sample_rois(all_rois, gt_boxes, gt_labels, fg_rois_per_image, rois_per_image, fg_thresh, bg_thresh_hi, bg_thresh_lo):
    """Generate a random sample of RoIs comprising foreground and background examples.
    gt_boxes is (1, N, 4) with 4 coordinates and 1 class label
    gt_labels is in one hot form
    """

    # overlaps: (rois x gt_boxes)
    overlaps = keras_rcnn.backend.overlap(all_rois, gt_boxes)
    gt_assignment = keras.backend.argmax(overlaps, axis=1)
    max_overlaps = keras.backend.max(overlaps, axis=1)
    labels = keras.backend.gather(gt_labels, gt_assignment)

    def no_sample(indices):
        return keras.backend.reshape(indices, (-1,))

    def sample(indices, size):
        return tensorflow.random_shuffle(keras.backend.reshape(indices, (-1,)))[:size]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = keras_rcnn.backend.where(max_overlaps >= fg_thresh)

    # Guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
    fg_rois_per_image = keras.backend.cast(fg_rois_per_image, 'int32')
    fg_rois_per_this_image = keras.backend.minimum(fg_rois_per_image, keras.backend.shape(fg_inds)[0])

    # Sample foreground regions without replacement
    fg_inds = tensorflow.cond(keras.backend.shape(fg_inds)[0] > 0, lambda: no_sample(fg_inds), lambda: sample(fg_inds, fg_rois_per_this_image))

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = keras_rcnn.backend.where((max_overlaps < bg_thresh_hi) & (max_overlaps >= bg_thresh_lo))

    # Compute number of background RoIs to take from this image (guarding against there being fewer than desired)
    rois_per_image = keras.backend.cast(rois_per_image, 'int32')
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = keras.backend.minimum(bg_rois_per_this_image, keras.backend.shape(bg_inds)[0])

    # Sample background regions without replacement
    bg_inds = tensorflow.cond(keras.backend.shape(bg_inds)[0] > 0, lambda: no_sample(bg_inds), lambda: sample(bg_inds, bg_rois_per_this_image))

    # The indices that we're selecting (both fg and bg)
    keep_inds = keras.backend.concatenate([fg_inds, bg_inds])

    # Select sampled values from various arrays:
    labels = keras.backend.gather(labels, keep_inds)

    # Clamp labels for the background RoIs to 0
    update_indices = keras.backend.arange(fg_rois_per_this_image, keras.backend.shape(labels)[0])
    update_indices_0 = keras.backend.reshape(update_indices, (-1, 1))
    update_indices_1 = keras_rcnn.backend.where(keras.backend.equal(keras.backend.gather(labels, update_indices), 1))[:, 1]
    update_indices_1 = keras.backend.reshape(keras.backend.cast(update_indices_1, 'int32'), (-1, 1))

    # By first removing the label
    update_indices = keras.backend.concatenate([update_indices_0, update_indices_1], axis=1)
    inverse_labels = tensorflow.gather_nd(labels, update_indices) * -1
    labels = keras_rcnn.backend.scatter_add_tensor(labels, update_indices, inverse_labels)

    # And then making the label = background
    update_indices = keras.backend.concatenate([update_indices_0, keras.backend.zeros_like(update_indices_0)], axis=1)
    inverse_labels = tensorflow.gather_nd(labels, update_indices) * -1
    labels = keras_rcnn.backend.scatter_add_tensor(labels, update_indices, inverse_labels + keras.backend.ones_like(inverse_labels))

    rois = keras.backend.gather(all_rois, keep_inds)

    # Compute bounding-box regression targets for an image.
    targets = keras_rcnn.backend.bbox_transform(rois, keras.backend.gather(gt_boxes, keras.backend.gather(gt_assignment, keep_inds)))
    bbox_targets = get_bbox_regression_labels(labels, targets)

    return rois, labels, bbox_targets
