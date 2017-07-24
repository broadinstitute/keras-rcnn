import keras.backend
import keras.layers
import keras_resnet.blocks


def residual(classes, mask=False):
    """Resnet classifiers as in Mask R-CNN."""
    def f(x):
        if keras.backend.image_data_format() == "channels_last":
            channel_axis = 3
        else:
            channel_axis = 1

        y = keras.layers.TimeDistributed(keras.layers.Conv2D(1024, (1, 1)))(x)

        # conv5 block as in Deep Residual Networks with first conv operates
        # on a 7x7 RoI with stride 1 (instead of 14x14 / stride 2)
        for i in range(3):
            y = keras_resnet.blocks.time_distributed_bottleneck_2d(512, (1, 1), first=True)(y)

        y = keras.layers.TimeDistributed(keras.layers.BatchNormalization(axis=channel_axis))(y)
        y = keras.layers.TimeDistributed(keras.layers.Activation("relu"))(y)

        # class and box branches
        y = keras.layers.TimeDistributed(keras.layers.AveragePooling2D((7, 7)))(y)

        score = keras.layers.TimeDistributed(keras.layers.Dense(classes, activation="softmax"))(y)

        boxes = keras.layers.TimeDistributed(keras.layers.Dense(4 * classes))(y)

        # TODO{JihongJu} the mask branch

        return [score, boxes]

    return f
