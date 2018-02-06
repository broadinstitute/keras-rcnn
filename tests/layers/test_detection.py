import keras.backend
import keras.utils
import numpy

import keras_rcnn.layers


# class TestDetection:
#     def test_call(self):
#         num_classes = 2
#         detections = keras_rcnn.layers.ObjectDetection()
#
#         proposals = numpy.array([[5, 10, 30, 50], [6.4, 1, 24.2, 33.2]])
#         proposals = keras.backend.variable(proposals)
#         pred_boxes = numpy.array([[.05, -.1, .03, .2, -.02, -.07, 0.1, -.04], [0.5, 0.3, -0.2, -0.4, .5, -.3, -.2, .4]])
#         pred_boxes = keras.backend.variable(pred_boxes)
#
#         pred_scores = numpy.array([[[0.4, 0.6], [0.9, .1]]])
#         pred_scores = keras.backend.variable(pred_scores)
#
#         metadata = keras.backend.variable([[35, 35, 1]])
#
#         boxes, classes = detections.call(
#             [proposals, pred_boxes, pred_scores, metadata], training=False)
#
#         # expected = numpy.array([[  5.90409106, 1.36124346, 32.69590894,  34.,
#         #                            3.11277807, 7.9338165, 31.84722193, 34.],
#         #                         [ 17.50393092,  16.43268724,  32.89606908,  34.,
#         #                           17.50393092,   0., 32.89606908,  32.40428998]])
#         expected = numpy.array([[ 3.11277807, 7.9338165, 31.84722193, 34.],
#                         [ 17.50393092,  16.43268724,  32.89606908,  34.]])
#         expected = numpy.expand_dims(expected, 0)
#
#         boxes = keras.backend.eval(boxes)
#         classes = keras.backend.eval(classes)
#
#         numpy.testing.assert_approx_equal(expected.sum(), boxes.sum(), 3)
#
#         assert boxes.shape == expected.shape
#         assert classes.shape[:2] == boxes.shape[:2]
#
#         assert classes.shape[-1] == num_classes
