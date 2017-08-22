import keras.backend
import numpy

import keras_rcnn.backend
import keras_rcnn.layers


class TestObjectProposal:
    def test_call(self):
        metadata = keras.backend.variable([[224, 224, 1.5]])

        deltas = numpy.random.random((1, 14, 14, 9 * 4))
        scores = numpy.random.random((1, 14, 14, 9 * 2))

        deltas = keras.backend.variable(deltas)
        scores = keras.backend.variable(scores)

        object_proposal = keras_rcnn.layers.ObjectProposal()

        object_proposal.call([metadata, deltas, scores])


def test_bbox_transform_inv():
    anchors = 9
    features = (14, 14)
    shifted = keras_rcnn.backend.shift(features, 16)
    deltas = numpy.zeros((features[0] * features[1] * anchors, 4))
    deltas = keras.backend.variable(deltas)
    pred_boxes = keras_rcnn.layers.object_detection._object_proposal.bbox_transform_inv(shifted, deltas)
    assert keras.backend.eval(pred_boxes).shape == (1764, 4)

    shifted = numpy.zeros((5, 4))
    deltas = numpy.reshape(numpy.arange(12*5), (5, -1))
    deltas = keras.backend.variable(deltas)
    pred_boxes = keras_rcnn.layers.object_detection._object_proposal.bbox_transform_inv(shifted, deltas)
    expected = numpy.array(
        [[ -3.19452805e+00,  -8.54276846e+00,   4.19452805e+00,
           1.15427685e+01,  -1.97214397e+02,  -5.42816579e+02,
           2.06214397e+02,   5.53816579e+02,  -1.10047329e+04,
          -2.99275709e+04,   1.10217329e+04,   2.99465709e+04],
        [ -6.01289642e+05,  -1.63449519e+06,   6.01314642e+05,
           1.63452219e+06,  -3.28299681e+07,  -8.92411330e+07,
           3.28300011e+07,   8.92411680e+07,  -1.79245640e+09,
          -4.87240170e+09,   1.79245644e+09,   4.87240174e+09],
        [ -9.78648047e+10,  -2.66024120e+11,   9.78648047e+10,
           2.66024120e+11,  -5.34323729e+12,  -1.45244248e+13,
           5.34323729e+12,   1.45244248e+13,  -2.91730871e+14,
          -7.93006726e+14,   2.91730871e+14,   7.93006726e+14],
        [ -1.59279659e+16,  -4.32967002e+16,   1.59279659e+16,
           4.32967002e+16,  -8.69637471e+17,  -2.36391973e+18,
           8.69637471e+17,   2.36391973e+18,  -4.74805971e+19,
          -1.29065644e+20,   4.74805971e+19,   1.29065644e+20],
        [ -2.59235276e+21,  -7.04674541e+21,   2.59235276e+21,
           7.04674541e+21,  -1.41537665e+23,  -3.84739263e+23,
           1.41537665e+23,   3.84739263e+23,  -7.72769468e+24,
          -2.10060520e+25,   7.72769468e+24,   2.10060520e+25]], dtype=numpy.float32)
    numpy.testing.assert_array_almost_equal(keras.backend.eval(pred_boxes)[0], expected[0], 0, verbose=True)

def test_filter_boxes():
    proposals = numpy.array(
        [[0, 2, 3, 10],
         [-1, -5, 4, 8],
         [0, 0, 1, 1]]
    )

    minimum = 3

    results = keras_rcnn.layers.object_detection._object_proposal.filter_boxes(proposals, minimum)

    numpy.testing.assert_array_equal(keras.backend.eval(results), numpy.array([0, 1]))
