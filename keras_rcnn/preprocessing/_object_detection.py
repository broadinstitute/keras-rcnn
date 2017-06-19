import random
import threading

import numpy
import numpy.random
import scipy.misc
import skimage.io


anchor_box_scales = [128, 256, 512]
anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]
im_size = 600
minimum = im_size
rpn_stride = 16
rpn_min_overlap = 0.3
rpn_max_overlap = 0.7


def union(au, bu):
    x = min(au[0], bu[0])
    y = min(au[1], bu[1])
    w = max(au[2], bu[2]) - x
    h = max(au[3], bu[3]) - y
    return x, y, w, h


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0, 0, 0, 0
    return x, y, w, h


def iou(a, b):
    # a and b should be (x1,y1,x2,y2)
    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    i = intersection(a, b)
    u = union(a, b)

    area_i = i[2] * i[3]
    area_u = u[2] * u[3]
    return float(area_i) / float(area_u)


def get_img_output_length(width, height):
    def get_output_length(input_length):
        # zero_pad
        input_length += 6
        # apply 4 strided convolutions
        filter_sizes = [7, 3, 1, 1]
        stride = 2
        for filter_size in filter_sizes:
            input_length = (input_length - filter_size + stride) // stride
        return input_length

    return get_output_length(width), get_output_length(height)


def anchor(image, shape, resized):
    width, height = shape
    resized_width, resized_height = resized

    downscale = float(rpn_stride)
    anchor_sizes = anchor_box_scales
    anchor_ratios = anchor_box_ratios
    num_anchors = len(anchor_sizes) * len(anchor_ratios)

    # calculate the output map size based on the network architecture
    (output_width, output_height) = get_img_output_length(resized_width, resized_height)

    n_anchratios = len(anchor_ratios)

    # initialise empty output objectives
    y_rpn_overlap = numpy.zeros((output_height, output_width, num_anchors))
    y_is_box_valid = numpy.zeros((output_height, output_width, num_anchors))
    y_rpn_regr = numpy.zeros((output_height, output_width, num_anchors * 4))

    num_bboxes = len(image['boxes'])

    num_anchors_for_bbox = numpy.zeros(num_bboxes).astype(int)
    best_anchor_for_bbox = -1 * numpy.ones((num_bboxes, 4)).astype(int)
    best_iou_for_bbox = numpy.zeros(num_bboxes).astype(numpy.float32)
    best_x_for_bbox = numpy.zeros((num_bboxes, 4)).astype(int)
    best_dx_for_bbox = numpy.zeros((num_bboxes, 4)).astype(numpy.float32)

    # get the GT box coordinates, and resize to account for image resizing
    gta = numpy.zeros((num_bboxes, 4))

    for bbox_num, bbox in enumerate(image['boxes']):
        # get the GT box coordinates, and resize to account for image resizing
        gta[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))
        gta[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
        gta[bbox_num, 2] = bbox['y1'] * (resized_height / float(height))
        gta[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))

    # rpn ground truth
    for anchor_size_idx in range(len(anchor_sizes)):
        for anchor_ratio_idx in range(n_anchratios):
            anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
            anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]

            for ix in range(output_width):
                # x-coordinates of the current anchor box
                x1_anc = downscale * (ix + 0.5) - anchor_x / 2
                x2_anc = downscale * (ix + 0.5) + anchor_x / 2

                # ignore boxes that go across image boundaries
                if x1_anc < 0 or x2_anc > resized_width:
                    continue

                for jy in range(output_height):
                    # y-coordinates of the current anchor box
                    y1_anc = downscale * (jy + 0.5) - anchor_y / 2
                    y2_anc = downscale * (jy + 0.5) + anchor_y / 2

                    # ignore boxes that go across image boundaries
                    if y1_anc < 0 or y2_anc > resized_height:
                        continue

                    # bbox_type indicates whether an anchor should be a target
                    bbox_type = 'neg'

                    # this is the best IOU for the (x,y) coord and the current anchor
                    # note that this is different from the best IOU for a GT bbox
                    best_iou_for_loc = 0.0

                    for bbox_num in range(num_bboxes):
                        # get IOU of the current GT box and the current anchor box
                        curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1_anc, y1_anc, x2_anc, y2_anc])

                        # calculate the regression targets if they will be needed
                        if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > rpn_max_overlap:
                            cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
                            cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
                            cxa = (x1_anc + x2_anc) / 2.0
                            cya = (y1_anc + y2_anc) / 2.0

                            tx = (cx - cxa) / (x2_anc - x1_anc)
                            ty = (cy - cya) / (y2_anc - y1_anc)
                            tw = numpy.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
                            th = numpy.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))

                        if image['boxes'][bbox_num]['class'] != 'bg':
                            # all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
                            if curr_iou > best_iou_for_bbox[bbox_num]:
                                best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
                                best_iou_for_bbox[bbox_num] = curr_iou
                                best_x_for_bbox[bbox_num, :] = [x1_anc, x2_anc, y1_anc, y2_anc]
                                best_dx_for_bbox[bbox_num, :] = [tx, ty, tw, th]

                            # we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
                            if curr_iou > rpn_max_overlap:
                                bbox_type = 'pos'
                                num_anchors_for_bbox[bbox_num] += 1

                                # we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
                                if curr_iou > best_iou_for_loc:
                                    best_iou_for_loc = curr_iou
                                    best_regr = (tx, ty, tw, th)

                            # if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
                            if rpn_min_overlap < curr_iou < rpn_max_overlap:
                                # gray zone between neg and pos
                                if bbox_type != 'pos':
                                    bbox_type = 'neutral'

                    # turn on or off outputs depending on IOUs
                    if bbox_type == 'neg':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif bbox_type == 'neutral':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif bbox_type == 'pos':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
                        y_rpn_regr[jy, ix, start:start + 4] = best_regr

    # we ensure that every bbox has at least one positive RPN region
    for idx in range(num_anchors_for_bbox.shape[0]):
        if num_anchors_for_bbox[idx] == 0:
            # no box with an IOU greater than zero ...
            if best_anchor_for_bbox[idx, 0] == -1:
                continue

            y_is_box_valid[
                best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], best_anchor_for_bbox[idx, 2] + n_anchratios *
                best_anchor_for_bbox[idx, 3]] = 1

            y_rpn_overlap[
                best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], best_anchor_for_bbox[idx, 2] + n_anchratios *
                best_anchor_for_bbox[idx, 3]] = 1

            start = 4 * (best_anchor_for_bbox[idx, 2] + n_anchratios * best_anchor_for_bbox[idx, 3])

            y_rpn_regr[best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], start:start + 4] = best_dx_for_bbox[idx, :]

    y_rpn_overlap = numpy.expand_dims(y_rpn_overlap, axis=0)

    y_is_box_valid = numpy.expand_dims(y_is_box_valid, axis=0)

    y_rpn_regr = numpy.expand_dims(y_rpn_regr, axis=0)

    pos_locs = numpy.where(numpy.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
    neg_locs = numpy.where(numpy.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

    num_pos = len(pos_locs[0])

    # one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
    # regions. We also limit it to 256 regions.
    num_regions = 256

    if len(pos_locs[0]) > num_regions / 2:
        val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions / 2)
        y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
        num_pos = num_regions / 2

    if len(neg_locs[0]) + num_pos > num_regions:
        val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
        y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

    y_rpn_cls = numpy.concatenate([y_is_box_valid, y_rpn_overlap], axis=-1)
    y_rpn_regr = numpy.concatenate([numpy.repeat(y_rpn_overlap, 4, axis=-1), y_rpn_regr], axis=-1)

    y1, y2 = numpy.copy(y_rpn_regr), numpy.copy(y_rpn_cls)

    y1 = numpy.transpose(y1, (0, 2, 1, 3))
    y2 = numpy.transpose(y2, (0, 2, 1, 3))

    return y1, y2


class _Iterator:
    def __init__(self, n, batch_size, shuffle, seed):
        self.batch_index = 0

        self.batch_size = batch_size

        self.index_generator = self._flow_index(n, batch_size, shuffle, seed)

        self.lock = threading.Lock()

        self.n = n

        self.shuffle = shuffle

        self.total_batches_seen = 0

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        self.reset()

        while True:
            if seed is not None:
                numpy.random.seed(seed + self.total_batches_seen)

            if self.batch_index == 0:
                index_array = numpy.arange(n)

                if shuffle:
                    index_array = numpy.random.permutation(n)

            current_index = (self.batch_index * batch_size) % n

            if n > current_index + batch_size:
                current_batch_size = batch_size

                self.batch_index += 1
            else:
                current_batch_size = n - current_index

                self.batch_index = 0

            self.total_batches_seen += 1

            yield index_array[current_index:current_index + current_batch_size], current_index, current_batch_size

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def next(self, *args, **kwargs):
        pass


class _DictionaryIterator(_Iterator):
    def __init__(self, dictionary, generator, shuffle=False, seed=None):
        self.dictionary = dictionary

        self.generator = generator

        _Iterator.__init__(self, len(dictionary), 1, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        index = index_array[0]

        pathname = self.dictionary[index]["filename"]

        image = skimage.io.imread(pathname)

        x, y, _ = self.dictionary[index]["shape"]

        if x <= y:
            resized = minimum, int((minimum / x) * y)
        else:
            resized = int((minimum / y) * x), minimum

        image = scipy.misc.imresize(image, resized, "bicubic")

        image = numpy.expand_dims(image, axis=0)

        scores, boxes = anchor(self.dictionary[index], (x, y), resized)

        return image, [boxes, scores]


class ObjectDetectionGenerator:
    def __init__(self):
        pass

    def flow(self, dictionary, shuffle=True, seed=None):
        return _DictionaryIterator(dictionary, self, shuffle=shuffle, seed=seed)
