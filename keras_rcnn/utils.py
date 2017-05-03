import itertools
import keras
import numpy


def propose(proposals, scales=None, stride=16, ratios=None):
    if ratios is None:
        ratios = [[1, 1], [1, 2], [2, 1]]

    if scales is None:
        scales = [128, 256, 512]

    boxes, scores = [], []

    if keras.backend.image_data_format() == "channels_last":
        (rows, columns) = proposals.shape[1:3]
    else:
        (rows, columns) = proposals.shape[2:]

    current = 0

    for ratio, scale in itertools.product(ratios, scales):
        anchor_x = (scale * ratio[0]) / stride
        anchor_y = (scale * ratio[1]) / stride

        if keras.backend.image_data_format() == "channels_last":
            proposal = proposals[0, :, :, current]

            score = scores[0, :, :, 4 * current:4 * current + 4]

            score = numpy.transpose(score, (2, 0, 1))
        else:
            proposal = proposals[0, current, :, :]

            score = scores[0, 4 * current:4 * current + 4, :, :]

        current += 1

        for row, column in itertools.product(rows, columns):
            if proposal[row, column] > 0.0:
                (tx, ty, tw, th) = score[:, row, column]

                x = column - anchor_x / 2
                y = row - anchor_y / 2

                cx = x + anchor_x / 2.0
                cy = y + anchor_y / 2.0

                cx1 = tx * anchor_x + cx
                cy1 = ty * anchor_y + cy

                w1 = numpy.exp(tw) * anchor_x
                h1 = numpy.exp(th) * anchor_y

                x1 = cx1 - w1 / 2.
                y1 = cy1 - h1 / 2.

                x1 = int(round(x1))
                y1 = int(round(y1))
                w1 = int(round(w1))
                h1 = int(round(h1))

                (x1, y1, w, h) = x1, y1, w1, h1

                x2 = x1 + max(1, w)
                y2 = y1 + max(1, h)

                x1 = max(x1, 0)
                y1 = max(y1, 0)

                x2 = min(x2, columns - 1)
                y2 = min(y2, rows - 1)

                if x2 - x1 < 1:
                    continue

                if y2 - y1 < 1:
                    continue

                boxes.append((x1, y1, x2, y2))

                scores.append(proposal[row, column])

    return numpy.array(boxes), numpy.array(scores)
