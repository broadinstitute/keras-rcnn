# -*- coding: utf-8 -*-

import keras.backend
import keras.layers
import tensorflow


class MaskRCNN(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MaskRCNN, self).__init__(**kwargs)

    def call(self, inputs, training=None, **kwargs):
        target_categories, target_masks, output_masks = inputs

        loss = self.compute_mask_loss(
            target_categories,
            target_masks,
            output_masks
        )

        self.add_loss(loss, inputs)

        return output_masks

    def compute_mask_loss(
            self,
            target_categories,
            target_masks,
            output_masks
    ):
        target_categories = keras.backend.argmax(target_categories)

        target_categories = keras.backend.squeeze(target_categories, 0)

        # FIXME: submit patch to Keras that adds axis paramter to `keras.backend.gather`.
        output_masks = tensorflow.gather(output_masks, target_categories, axis=-1)[:, :, :, :, 0]

        loss = keras.backend.binary_crossentropy(target_masks, output_masks)

        return keras.backend.mean(loss)

    def compute_output_shape(self, input_shape):
        return input_shape[1]
