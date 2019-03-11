import keras_rcnn.layers.losses
import keras.backend
import numpy

import pytest


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    predictions = numpy.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -numpy.sum(targets * numpy.log(predictions))
    return ce


class TestMaskRCNN:

    def test_categorical_crossentropy(self):
        layer = keras_rcnn.layers.losses.RCNNMaskLoss()

        n1 = 30
        n2 = 50
        N = 10

        output = numpy.random.random((n1, N))
        target = numpy.random.random((n2, N))
        output_tensor = keras.backend.tf.convert_to_tensor(output, dtype=keras.backend.tf.float32)
        target_tensor = keras.backend.tf.convert_to_tensor(target, dtype=keras.backend.tf.float32)

        expected_losses = numpy.zeros((n2, n1))
        for i in range(0, n2):
            for j in range(0, n1):
                expected_losses[i, j] = cross_entropy(output[j, :], target[i, :], epsilon=keras.backend.epsilon())

        losses = numpy.array(keras.backend.eval(layer.categorical_crossentropy(target=target_tensor, output=output_tensor)))

        precision = 0.001
        expected_loss = round(numpy.mean(expected_losses)/precision)*precision
        loss = round(numpy.mean(losses)/precision)*precision

        # print('')
        # print(loss)
        # print(expected_loss)

        assert expected_loss == loss

    def test_binary_crossentropy(self):
        layer = keras_rcnn.layers.losses.RCNNMaskLoss()

        n1 = 30
        n2 = 50

        output = numpy.random.random((n1, 1))
        output = numpy.concatenate((output, 1-output), axis=1)
        target = numpy.random.random((n2, 1))
        target = numpy.concatenate((target, 1 - target), axis=1)
        output_tensor = keras.backend.tf.convert_to_tensor(output, dtype=keras.backend.tf.float32)
        target_tensor = keras.backend.tf.convert_to_tensor(target, dtype=keras.backend.tf.float32)

        lossesBCE = numpy.array(
            keras.backend.eval(layer.binary_crossentropy(target=target_tensor, output=output_tensor)))

        lossesCCE = numpy.array(
            keras.backend.eval(layer.categorical_crossentropy(target=target_tensor, output=output_tensor)))


        precision = 1000
        lossBCE = round(numpy.mean(lossesBCE)*precision) / precision
        lossCCE = round(numpy.mean(lossesCCE)*precision) / precision

        # print('')
        # print(lossBCE)
        # print(lossCCE)

        assert lossBCE == lossCCE
        # assert True

    def test_compute_mask_loss(self):
        layer = keras_rcnn.layers.losses.RCNNMaskLoss()

        threshold = 0.15

        nbTar = 50
        nbOut = 30
        mask_size = 28
        target_bounding_boxes1 = numpy.random.randint(0, 256, (1, nbTar, 1))
        target_bounding_boxes2 = numpy.random.randint(256, 512, (1, nbTar, 1))
        target_bounding_boxes3 = numpy.random.randint(0, 256, (1, nbTar, 1))
        target_bounding_boxes4 = numpy.random.randint(256, 512, (1, nbTar, 1))

        bb_true = numpy.concatenate((
            target_bounding_boxes1,
            target_bounding_boxes2,
            target_bounding_boxes3,
            target_bounding_boxes4), axis=2)
        target_bounding_boxes = keras.backend.tf.convert_to_tensor(bb_true, dtype=keras.backend.tf.float32)

        output_bounding_boxes1 = numpy.random.randint(0, 256, (1, nbOut, 1))
        output_bounding_boxes2 = numpy.random.randint(256, 512, (1, nbOut, 1))
        output_bounding_boxes3 = numpy.random.randint(0, 256, (1, nbOut, 1))
        output_bounding_boxes4 = numpy.random.randint(256, 512, (1, nbOut, 1))

        bb_pred = numpy.concatenate((
            output_bounding_boxes1,
            output_bounding_boxes2,
            output_bounding_boxes3,
            output_bounding_boxes4), axis=2)
        output_bounding_boxes = keras.backend.tf.convert_to_tensor(bb_pred, dtype=keras.backend.tf.float32)

        y_true = numpy.random.randint(0, 2, (1, nbTar, mask_size, mask_size)).astype(float)
        target_masks = keras.backend.tf.convert_to_tensor(y_true, dtype=keras.backend.tf.float32)
        y_pred = numpy.random.random((1, nbOut, mask_size, mask_size))
        output_masks = keras.backend.tf.convert_to_tensor(y_pred, dtype=keras.backend.tf.float32)

        loss = layer.compute_mask_loss(
            target_bounding_box=target_bounding_boxes,
            output_bounding_box=output_bounding_boxes,
            target_mask=target_masks,
            output_mask=output_masks,
            threshold=threshold)

        loss = keras.backend.eval(loss)

        bb_pred = numpy.squeeze(bb_pred, axis=0)
        bb_true = numpy.squeeze(bb_true, axis=0)
        count = 0.0
        for i in range(0, nbTar):
            for j in range(0, nbOut):
                iou = bb_intersection_over_union(bb_true[i,:], bb_pred[j,:])
                if iou > threshold:
                    tm = keras.backend.eval(target_masks[0,i,:,:])
                    om = keras.backend.eval(output_masks[0,j,:,:])
                    tm = keras.backend.tf.convert_to_tensor(numpy.array([tm.flatten()]))
                    om = keras.backend.tf.convert_to_tensor(numpy.array([om.flatten()]))
                    cce = keras.backend.eval(layer.binary_crossentropy(target=tm,output=om))[0, 0]
                    count += cce


        expected_loss = count/nbTar/nbOut

        precision = 1000000.0
        loss = round(loss*precision)
        expected_loss = round(expected_loss*precision)

        # print('')
        # print(loss)
        # print(expected_loss)

        assert loss == expected_loss

    def test_intersection_over_union(self):
        s1 = 50
        s2 = 30
        x = numpy.random.random((s1, 4))
        y = numpy.random.random((s2, 4))
        layer = keras_rcnn.layers.losses.RCNNMaskLoss()

        iou1 = numpy.array(keras.backend.eval(layer.intersection_over_union(x, y)))
        iou2 = numpy.zeros((s1, s2))

        for i in range(0, s1):
            for j in range(0, s2):
                iou2[i,j] = bb_intersection_over_union(x[i,:], y[j,:])

        # print('')
        # print(numpy.sum(iou1 != iou2))
        assert numpy.sum(iou1 != iou2) == 0

    def test_layer(self):
        print('')
        layer = keras_rcnn.layers.losses.RCNNMaskLoss()

        # TESTS CROSS-ENTROPY ------------------------------------------------------------------------------------------
        n1 = 30
        n2 = 50
        N = 10

        output = numpy.random.random((n1, N))
        target = numpy.random.random((n2, N+1))
        output_tensor = keras.backend.tf.convert_to_tensor(output, dtype=keras.backend.tf.float32)
        target_tensor = keras.backend.tf.convert_to_tensor(target, dtype=keras.backend.tf.float32)

        with pytest.raises(Exception) as e_info:
            layer.categorical_crossentropy(target=target_tensor, output=output_tensor)
        print(e_info)
        with pytest.raises(Exception):
            layer.binary_crossentropy(target=target_tensor, output=output_tensor)
        print(e_info)

        # TEST SIZE TARGET ---------------------------------------------------------------------------------------------
        nb1 = 20
        nb2 = 40
        N = 15
        M = 25

        tbb = numpy.random.random((1, nb1, 4))
        obb = numpy.random.random((1, nb2, 4))
        tm = numpy.random.random((1, nb1 + 1, N, M))
        om = numpy.random.random((1, nb2, N, M))
        target_bounding_box = keras.backend.tf.convert_to_tensor(tbb, dtype=keras.backend.tf.float32)
        output_bounding_box = keras.backend.tf.convert_to_tensor(obb, dtype=keras.backend.tf.float32)
        target_mask = keras.backend.tf.convert_to_tensor(tm, dtype=keras.backend.tf.float32)
        output_mask = keras.backend.tf.convert_to_tensor(om, dtype=keras.backend.tf.float32)

        with pytest.raises(Exception) as e_info:
            layer.compute_mask_loss(target_bounding_box=target_bounding_box,
                                    output_bounding_box=output_bounding_box,
                                    target_mask=target_mask,
                                    output_mask=output_mask)
        print(e_info)

        # TEST SIZE OUTPUT ---------------------------------------------------------------------------------------------
        tm = numpy.random.random((1, nb1, N, M))
        om = numpy.random.random((1, nb2 + 1, N, M))
        target_mask = keras.backend.tf.convert_to_tensor(tm, dtype=keras.backend.tf.float32)
        output_mask = keras.backend.tf.convert_to_tensor(om, dtype=keras.backend.tf.float32)

        with pytest.raises(Exception) as e_info:
            layer.compute_mask_loss(target_bounding_box=target_bounding_box,
                                    output_bounding_box=output_bounding_box,
                                    target_mask=target_mask,
                                    output_mask=output_mask)
        print(e_info)

        # TESTS PARAMS BOUNDING BOXES ----------------------------------------------------------------------------------
        tbb = numpy.random.random((1, nb1, 3))
        obb = numpy.random.random((1, nb2, 4))
        tm = numpy.random.random((1, nb1, N, M))
        om = numpy.random.random((1, nb2, N, M))
        target_bounding_box = keras.backend.tf.convert_to_tensor(tbb, dtype=keras.backend.tf.float32)
        output_bounding_box = keras.backend.tf.convert_to_tensor(obb, dtype=keras.backend.tf.float32)
        target_mask = keras.backend.tf.convert_to_tensor(tm, dtype=keras.backend.tf.float32)
        output_mask = keras.backend.tf.convert_to_tensor(om, dtype=keras.backend.tf.float32)

        with pytest.raises(Exception) as e_info:
            layer.compute_mask_loss(target_bounding_box=target_bounding_box,
                                    output_bounding_box=output_bounding_box,
                                    target_mask=target_mask,
                                    output_mask=output_mask)
        print(e_info)

        tbb = numpy.random.random((1, nb1, 4))
        obb = numpy.random.random((1, nb2, 3))
        target_bounding_box = keras.backend.tf.convert_to_tensor(tbb, dtype=keras.backend.tf.float32)
        output_bounding_box = keras.backend.tf.convert_to_tensor(obb, dtype=keras.backend.tf.float32)

        with pytest.raises(Exception) as e_info:
            layer.compute_mask_loss(target_bounding_box=target_bounding_box,
                                    output_bounding_box=output_bounding_box,
                                    target_mask=target_mask,
                                    output_mask=output_mask)
        print(e_info)

        tbb = numpy.random.random((1, nb1, 3))
        obb = numpy.random.random((1, nb2, 3))
        target_bounding_box = keras.backend.tf.convert_to_tensor(tbb, dtype=keras.backend.tf.float32)
        output_bounding_box = keras.backend.tf.convert_to_tensor(obb, dtype=keras.backend.tf.float32)

        with pytest.raises(Exception) as e_info:
            layer.compute_mask_loss(target_bounding_box=target_bounding_box,
                                    output_bounding_box=output_bounding_box,
                                    target_mask=target_mask,
                                    output_mask=output_mask)
        print(e_info)

        # TEST BATCH DIMENSION -----------------------------------------------------------------------------------------
        tbb = numpy.random.random((nb1, 4))
        obb = numpy.random.random((nb2, 4))
        tm = numpy.random.random((nb1, N, M))
        om = numpy.random.random((nb2, N, M))
        target_bounding_box = keras.backend.tf.convert_to_tensor(tbb, dtype=keras.backend.tf.float32)
        output_bounding_box = keras.backend.tf.convert_to_tensor(obb, dtype=keras.backend.tf.float32)
        target_mask = keras.backend.tf.convert_to_tensor(tm, dtype=keras.backend.tf.float32)
        output_mask = keras.backend.tf.convert_to_tensor(om, dtype=keras.backend.tf.float32)

        with pytest.raises(Exception) as e_info:
            layer.compute_mask_loss(target_bounding_box=target_bounding_box,
                                    output_bounding_box=output_bounding_box,
                                    target_mask=target_mask,
                                    output_mask=output_mask)
        print(e_info)

        # TEST MASK FEATURES DIMENSION ---------------------------------------------------------------------------------
        tbb = numpy.random.random((1, nb1, 4))
        obb = numpy.random.random((1, nb2, 4))
        tm = numpy.random.random((1, nb1, N, M))
        om = numpy.random.random((1, nb2, N, M, 2))
        target_bounding_box = keras.backend.tf.convert_to_tensor(tbb, dtype=keras.backend.tf.float32)
        output_bounding_box = keras.backend.tf.convert_to_tensor(obb, dtype=keras.backend.tf.float32)
        target_mask = keras.backend.tf.convert_to_tensor(tm, dtype=keras.backend.tf.float32)
        output_mask = keras.backend.tf.convert_to_tensor(om, dtype=keras.backend.tf.float32)

        with pytest.raises(Exception) as e_info:
            layer.compute_mask_loss(target_bounding_box=target_bounding_box,
                                    output_bounding_box=output_bounding_box,
                                    target_mask=target_mask,
                                    output_mask=output_mask)
        print(e_info)

        assert True
