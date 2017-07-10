import keras.layers
import keras_rcnn.models


def test_resnet50_rcnn():
    inputs = keras.layers.Input((224, 224, 3))

    model = keras_rcnn.models.ResNet50RCNN(inputs, 21, 300)

    model.compile(loss=["mse", "mse", "mse"], optimizer="adam")
