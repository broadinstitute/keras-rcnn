import keras.backend
import keras_rcnn.layers
import numpy


class TestRCNN():
    def test_call(self):
        classes = 3

        target_deltas = keras.backend.ones((1, 2, 4 * classes))
        target_scores = keras.backend.variable([[0, 0, 1], [0, 0, 1]])
        target_scores = keras.backend.expand_dims(target_scores, 0)
        target_scores = keras.backend.cast(target_scores, keras.backend.floatx())

        output_deltas = keras.backend.ones((1, 2, 4 * classes))
        output_scores = keras.backend.variable([[0, 0, 1], [0, 0, 1]])
        output_scores = keras.backend.expand_dims(output_scores, 0)
        output_scores = keras.backend.cast(output_scores, keras.backend.floatx())

        layer = keras_rcnn.layers.RCNN()

        layer.get_config()

        layer.call([target_deltas, target_scores, output_deltas, output_scores])

        classification_loss = layer.classification_loss()

        classification_loss = keras.backend.eval(classification_loss)

        numpy.testing.assert_almost_equal(classification_loss, 0.0)

        regression_loss = layer.regression_loss()

        regression_loss = keras.backend.eval(regression_loss)

        numpy.testing.assert_almost_equal(regression_loss, 0.0)

        loss = layer.losses.pop()

        loss = keras.backend.eval(loss)

        numpy.testing.assert_almost_equal(loss, 0.0)

        target_scores = keras.backend.variable([[0, 0, 1], [0, 0, 1]])
        target_scores = keras.backend.expand_dims(target_scores, 0)

        output_scores = keras.backend.variable([[0, 1, 0], [0, 1, 0]])
        output_scores = keras.backend.expand_dims(output_scores, 0)

        layer.call([target_deltas, target_scores, output_deltas, output_scores])

        classification_loss = layer.classification_loss()

        classification_loss = keras.backend.eval(classification_loss)

        numpy.testing.assert_almost_equal(classification_loss, numpy.log(1.0 / keras.backend.epsilon()), 5)

        regression_loss = layer.regression_loss()

        regression_loss = keras.backend.eval(regression_loss)

        numpy.testing.assert_almost_equal(regression_loss, 0.0)

        loss = layer.losses.pop()

        loss = keras.backend.eval(loss)

        numpy.testing.assert_almost_equal(loss, numpy.log(1.0 / keras.backend.epsilon()), 5)

        target_scores = keras.backend.variable([[0, 0, 1], [0, 1, 0]])
        target_scores = keras.backend.expand_dims(target_scores, 0)
        output_scores = keras.backend.variable([[0, 1, 0], [0, 1, 0]])
        output_scores = keras.backend.expand_dims(output_scores, 0)

        target_deltas = keras.backend.variable([[0, 0, 0, 0, 0, 0, 0, 0, -.1, .2, .3, .4], [-.1, .2, 1, -3, 0, -1, 1.3, -.1, 2, .3, -1.5, .6]])
        target_deltas = keras.backend.expand_dims(target_deltas, 0)
        output_deltas = keras.backend.variable([[-.1, .2, 1, -3, 0, 0, .1, 1, 0, .1, -.1, .8], [-.1, 2, 0, 0, 1, 1, -.1, -.5, 0, 1, -.5, -1]])
        output_deltas = keras.backend.expand_dims(output_deltas, 0)

        layer.call([target_deltas, target_scores, output_deltas, output_scores])

        regression_loss = layer.regression_loss()

        regression_loss = keras.backend.eval(regression_loss)

        numpy.testing.assert_almost_equal(regression_loss, 1.575)
