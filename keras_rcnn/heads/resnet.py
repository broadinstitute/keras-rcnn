import keras


def ResHead(classes, mask=False):
    """Resnet heads as in Mask R-CNN."""
    def f(x):
        if keras.backend.image_data_format() == "channels_last":
            channel_axis = 3
        else:
            channel_axis = 1

        y = keras.layers.TimeDistributed(
            keras.layers.Conv2D(1024, (1, 1)))(x)

        # conv5 block as in Deep Residual Networks with first conv operates
        # on a 7x7 RoI with stride 1 (instead of 14x14 / stride 2)
        for i in range(3):
            y = _bottleneck(512, (1, 1))(y)

        y = keras.layers.TimeDistributed(
            keras.layers.BatchNormalization(axis=channel_axis))(y)
        y = keras.layers.TimeDistributed(
            keras.layers.Activation("relu"))(y)

        # class and box branches
        y = keras.layers.TimeDistributed(
            keras.layers.AveragePooling2D((7, 7)))(y)

        score = keras.layers.TimeDistributed(
            keras.layers.Dense(classes, activation="softmax"))(y)

        boxes = keras.layers.TimeDistributed(
            keras.layers.Dense(4 * classes))(y)

        # TODO{JihongJu} the mask branch

        return [score, boxes]
    return f


def _bottleneck(filters, strides=(1, 1)):
    """Time Distributed bottleneck block."""
    def f(x):
        if keras.backend.image_data_format() == "channels_last":
            channel_axis = 3
        else:
            channel_axis = 1
        y = keras.layers.TimeDistributed(
            keras.layers.Conv2D(filters, (1, 1), strides=strides,
                                padding="same"))(x)

        y = keras.layers.TimeDistributed(
            keras.layers.BatchNormalization(axis=channel_axis))(y)
        y = keras.layers.TimeDistributed(keras.layers.Activation("relu"))(y)

        y = keras.layers.TimeDistributed(keras.layers.Conv2D(
            filters, (3, 3), padding="same"))(y)

        y = keras.layers.TimeDistributed(
            keras.layers.BatchNormalization(axis=channel_axis))(y)
        y = keras.layers.TimeDistributed(keras.layers.Activation("relu"))(y)

        y = keras.layers.TimeDistributed(
            keras.layers.Conv2D(filters * 4, (1, 1)))(y)

        y = keras.layers.TimeDistributed(
            keras.layers.BatchNormalization(axis=channel_axis))(y)
        y = _shortcut(x, y)
        y = keras.layers.TimeDistributed(keras.layers.Activation("relu"))(y)

        return y

    return f


def _shortcut(a, b):
    """Time Distributed shortcut."""
    if keras.backend.image_data_format() == "channels_last":
        row_axis, col_axis, channel_axis = 1, 2, 3
    else:
        row_axis, col_axis, channel_axis = 2, 3, 4
    a_shape = keras.backend.int_shape(a)[1:]
    b_shape = keras.backend.int_shape(b)[1:]

    x = int(round(a_shape[row_axis] / b_shape[row_axis]))
    y = int(round(a_shape[col_axis] / b_shape[col_axis]))

    if x > 1 or y > 1 or not a_shape[channel_axis] == b_shape[channel_axis]:
        a = keras.layers.TimeDistributed(
            keras.layers.Conv2D(b_shape[channel_axis], (1, 1), strides=(x, y),
                                padding="same"))(a)

        a = keras.layers.TimeDistributed(
            keras.layers.BatchNormalization(axis=channel_axis))(a)

    return keras.layers.add([a, b])
