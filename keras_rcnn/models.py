import keras.layers
import keras.models


class FasterRCNN(keras.models.Model):
    def __init__(self, inputs, outputs):
        super(FasterRCNN, self).__init__(inputs, outputs)


class RPN(keras.models.Model):
    def __init__(self, inputs, anchors):
        y = keras.layers.Conv2D(512, (3, 3), padding='same')(inputs)
        y = keras.layers.Activation("relu")(y)

        classifier = keras.layers.Conv2D(anchors, (1, 1))(y)
        classifier = keras.layers.Activation("sigmoid")(classifier)

        regression = keras.layers.Conv2D(anchors * 4, (1, 1))(y)
        regression = keras.layers.Activation("linear")(regression)

        super(RPN, self).__init__(inputs, [classifier, regression])
