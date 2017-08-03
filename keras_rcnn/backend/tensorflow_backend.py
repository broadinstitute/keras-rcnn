import keras.backend
import tensorflow
import keras_rcnn.backend

RPN_FG_FRACTION = 0.5
RPN_BATCHSIZE = 256

#TODO: remove globals
def argsort(a, axis=-1):
    _, indices = tensorflow.nn.top_k(a, keras.backend.shape(a)[-1])

    return indices

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
    return tensorflow.image.non_max_suppression(
        boxes=boxes,
        iou_threshold=threshold,
        max_output_size=maximum,
        scores=scores
    )


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
    reference = keras_rcnn.backend.overlap(anchors, gt_boxes[:, :4])

    argmax_overlaps_inds = keras.backend.argmax(reference, axis=1)

    gt_argmax_overlaps_inds = keras.backend.argmax(reference, axis=0)

    indices = keras.backend.stack([
        tensorflow.range(keras.backend.shape(inds_inside)[0]),
        keras.backend.cast(argmax_overlaps_inds, "int32")
    ], axis=0)

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

    num_fg = RPN_FG_FRACTION * RPN_BATCHSIZE

    fg_inds = keras.backend.shape(keras_rcnn.backend.where(keras.backend.equal(labels, 1)))[0]

    size = tensorflow.subtract(tensorflow.cast(fg_inds, tensorflow.int32), tensorflow.cast(num_fg, tensorflow.int32))

    def more_positive():
        elems = tensorflow.multinomial(tensorflow.log(tensorflow.ones((1, fg_inds))), size)

        indices = tensorflow.reshape(elems, (-1, 1))

        return scatter_add_tensor(labels, indices, tensorflow.ones((size,)) * -1)

    def less_positive():
        return labels

    predicate = tensorflow.less_equal(size, 0)

    return tensorflow.cond(predicate, lambda: less_positive(), lambda: more_positive())


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


def label(y_true, y_pred, inds_inside, RPN_NEGATIVE_OVERLAP=0.3, RPN_POSITIVE_OVERLAP=0.7, clobber_positives=True):
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

    argmax_overlaps_inds, max_overlaps, gt_argmax_overlaps_inds = overlapping(y_true, y_pred, inds_inside)

    # assign bg labels first so that positive labels can clobber them
    if not clobber_positives:
        labels = keras_rcnn.backend.where(keras.backend.less(max_overlaps, RPN_NEGATIVE_OVERLAP), zeros, labels)

    # fg label: for each gt, anchor with highest overlap
    indices = keras.backend.expand_dims(gt_argmax_overlaps_inds, axis=1)

    updates = keras.backend.ones_like(gt_argmax_overlaps_inds, dtype=keras.backend.floatx())
    labels = keras_rcnn.backend.scatter_add_tensor(labels, indices, updates)

    # fg label: above threshold IOU
    labels = keras_rcnn.backend.where(keras.backend.greater_equal(max_overlaps, RPN_POSITIVE_OVERLAP), ones, labels)

    if clobber_positives:
        # assign bg labels last so that negative labels can clobber positives
        labels = keras_rcnn.backend.where(keras.backend.less(max_overlaps, RPN_NEGATIVE_OVERLAP), zeros, labels)

    return argmax_overlaps_inds, balance(labels)

