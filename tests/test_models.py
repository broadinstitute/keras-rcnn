import keras.layers
import keras_rcnn.models


def test_resnet50_rcnn():
    inputs = keras.layers.Input((223, 223, 3))
    model = keras_rcnn.models.RCNN(inputs, 21, 300)
    model.compile(loss=["mse", "mse", "mse"],
                  optimizer="adam")
