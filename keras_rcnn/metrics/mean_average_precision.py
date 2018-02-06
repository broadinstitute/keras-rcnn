import numpy

def average_precision(tp, fp, n):
    """

    Args:
        tp: (d, ) for each detection, whether it's a true positive
        fp: (d, ) for each detection, whether it's a false positive
        n: number of positive examples

    Returns: average precision for binary classification

    """
    fp = numpy.cumsum(fp)
    tp = numpy.cumsum(tp)

    r = tp / float(n)

    epsilon = numpy.finfo(numpy.float32).eps

    p = tp / numpy.maximum(tp + fp, epsilon)

    r = numpy.concatenate(([0.0], r, [1.0]))
    p = numpy.concatenate(([0.0], p, [0.0]))

    for index in range(p.size - 1, 0, -1):
        p[index - 1] = numpy.maximum(p[index - 1], p[index])

    indices = numpy.where(r[1:] != r[:-1])[0]

    return numpy.sum((r[indices + 1] - r[indices]) * p[indices + 1])

def intersection_over_union(target, output):
    """
    :param target: (m, 4)
    :param output: (n, 4)

    :return: (m × n, 1)
    """
    target_x1, target_y1, target_x2, target_y2 = numpy.split(target, 4, axis=1)
    output_x1, output_y1, output_x2, output_y2 = numpy.split(output, 4, axis=1)

    intersection_x1 = numpy.maximum(target_x1, numpy.transpose(output_x1))
    intersection_y1 = numpy.maximum(target_y1, numpy.transpose(output_y1))

    intersection_x2 = numpy.minimum(target_x2, numpy.transpose(output_x2))
    intersection_y2 = numpy.minimum(target_y2, numpy.transpose(output_y2))

    intersection_cc = numpy.maximum(0, intersection_x2 - intersection_x1 + 1)
    intersection_rr = numpy.maximum(0, intersection_y2 - intersection_y1 + 1)

    intersection = intersection_cc * intersection_rr

    target_union_cc = target_x2 - target_x1 + 1
    target_union_rr = target_y2 - target_y1 + 1

    output_union_cc = output_x2 - output_x1 + 1
    output_union_rr = output_y2 - output_y1 + 1

    target_union = target_union_cc * target_union_rr
    output_union = output_union_cc * output_union_rr

    union = target_union + numpy.transpose(output_union)

    return numpy.maximum(0.0, intersection / (union - intersection))

def mean_average_precision(target, output, class_descriptions):
    """

    Args:
        target: ground truth
        output: prediction output [boxes (batch size, detections, 4), scores (batch size, detections, classes)]
        class_descriptions: list of class names, where class_descriptions['background'] = 0

    Returns: mean average precision

    """
    output_bounding_boxes, output_all_scores = output

    # NOTE: The image indices for each detection.

    image_indices = numpy.reshape([index for index in range(len(output_all_scores)) for i in range(len(output_all_scores[index]))], output_all_scores.shape[:2])
    
    # concatenate/reshape 
    output_bounding_boxes = numpy.reshape(output_bounding_boxes, (-1, 4))
    output_all_scores = numpy.reshape(output_all_scores, (-1, len(class_descriptions)+1))
    image_indices = numpy.reshape(image_indices, (-1,))

    # remove filler detections
    filler = numpy.sum(output_all_scores, axis = -1) > 0
    output_all_scores = output_all_scores[filler]
    output_bounding_boxes = output_bounding_boxes[filler]
    image_indices = image_indices[filler]

    aps = []

    threshold = 0.5

    for class_name in class_descriptions:

        class_index = class_descriptions[class_name]

        if class_index == 0:
            continue

        detections_per_class = {}

        number_of_instances_per_class = 0

        output_scores = output_all_scores[..., class_index]

        for index, target_image in enumerate(target):
            instances = [instance for instance in target_image["boxes"] if instance["class"] == class_name]

            target_bounding_boxes = numpy.array([(instance["x1"], instance["y1"], instance["x2"], instance["y2"]) for instance in instances])

            difficult = numpy.array([instance["class"] == "difficult" for instance in instances]).astype(numpy.bool)

            # NOTE: `detected` tracks whether an instance was detected
            detected = [False] * len(instances)

            number_of_easy_instances = sum(~difficult)

            number_of_instances_per_class += number_of_easy_instances

            detections_per_class[index] = {
                "boxes": target_bounding_boxes,
                "difficult": difficult,
                "detected": detected
            }


        # sort by score
        sorted_indices = numpy.argsort(-output_scores)
        bounding_boxes = output_bounding_boxes[sorted_indices]
        sorted_image_indices = image_indices[sorted_indices]

        number_of_detections = len(sorted_image_indices)
        tp = numpy.zeros(number_of_detections)
        fp = numpy.zeros(number_of_detections)

        for detection_index in range(number_of_detections):

            instances = detections_per_class[sorted_image_indices[detection_index]]
            instance_bounding_box = bounding_boxes[detection_index]

            maximum_overlap_ratio = 0.0

            instance_target_bounding_boxes = instances["boxes"]

            # If there’s a target bounding box:

            if instance_target_bounding_boxes.size > 0:
                instance_bounding_box = numpy.expand_dims(instance_bounding_box, 0)

                overlap_ratios = intersection_over_union(instance_target_bounding_boxes, instance_bounding_box)

                maximum_overlap_index = numpy.argmax(overlap_ratios)

                maximum_overlap_ratio = numpy.squeeze(overlap_ratios[maximum_overlap_index])

            if maximum_overlap_ratio > threshold:

                if not instances["difficult"][maximum_overlap_index]:
                    if not instances["detected"][maximum_overlap_index]:
                        tp[detection_index] = 1.0
                        instances["detected"][maximum_overlap_index] = 1
                    else:
                        fp[detection_index] = 1.0
            else:
                fp[detection_index] = 1.0
        ap = average_precision(tp, fp, number_of_instances_per_class)
        aps.append(ap)
    return numpy.nanmean(aps), aps
