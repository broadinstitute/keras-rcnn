import keras.backend
import keras_rcnn.layers
import numpy

class TestMaskRCNN():
    def test_call(self):
        layer = keras_rcnn.layers.MaskRCNN()

        target_categories = numpy.expand_dims([[0, 1]], 0)

        target_masks = numpy.zeros((1, 1, 28, 28))

        output_masks = numpy.zeros((1, 1, 28, 28, 2))

        layer.call([target_categories, target_masks, output_masks])

        mask_loss = layer.compute_mask_loss(target_categories, target_masks, output_masks)

        mask_loss = keras.backend.eval(mask_loss)

        target_categories = numpy.expand_dims([[0, 0, 1]], 0)

        numpy.testing.assert_almost_equal(mask_loss, 0.0)

        output_masks_1 = numpy.zeros((1, 1, 28, 28, 1))

        output_masks_2 = numpy.ones((1, 1, 28, 28, 1))

        output_masks_3 = numpy.ones((1, 1, 28, 28, 1)) * 2

        output_masks = numpy.concatenate((output_masks_1, output_masks_2, output_masks_3), axis=-1)

        layer.call([target_categories, target_masks, output_masks])

        mask_loss = layer.compute_mask_loss(target_categories, target_masks, output_masks)

        mask_loss = keras.backend.eval(mask_loss)

        numpy.testing.assert_almost_equal(mask_loss, numpy.log(1.0 / keras.backend.epsilon()), 0), mask_loss

