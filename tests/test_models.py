import keras.layers
import keras_rcnn.models
import keras.layers
import keras.backend
import keras.models
import numpy
import keras_rcnn.losses.rpn
import keras_rcnn.models


def test_resnet50_rcnn():
    inputs = keras.layers.Input((224, 224, 3))
    model = keras_rcnn.models.ResNet50RCNN(inputs, 21, 300)
    model.compile(loss=["mse", "mse", "mse"],
                  optimizer="adam")

def test_rpn():
    options = {
        "activation": "relu",
        "kernel_size": (3, 3),
        "padding": "same"
    }

    image = keras.layers.Input((224, 224, 3))

    y_true = keras.layers.Input((None, 4), name="y_true")


    features = keras.layers.Conv2D(64, **options)(image)
    features = keras.layers.Conv2D(64, **options)(features)

    features = keras.layers.MaxPooling2D(strides=(2, 2))(features)

    features = keras.layers.Conv2D(128, **options)(features)
    features = keras.layers.Conv2D(128, **options)(features)

    features = keras.layers.MaxPooling2D(strides=(2, 2))(features)

    features = keras.layers.Conv2D(256, **options)(features)
    features = keras.layers.Conv2D(256, **options)(features)
    features = keras.layers.Conv2D(256, **options)(features)
    features = keras.layers.Conv2D(256, **options)(features)

    features = keras.layers.MaxPooling2D(strides=(2, 2))(features)

    features = keras.layers.Conv2D(512, **options)(features)
    features = keras.layers.Conv2D(512, **options)(features)
    features = keras.layers.Conv2D(512, **options)(features)
    features = keras.layers.Conv2D(512, **options)(features)

    features = keras.layers.MaxPooling2D(strides=(2, 2))(features)

    features = keras.layers.Conv2D(512, **options)(features)
    features = keras.layers.Conv2D(512, **options)(features)
    features = keras.layers.Conv2D(512, **options)(features)

    features = keras.layers.Conv2D(512, **options)(features)

    classification = keras.layers.Conv2D(9 * 1, (1, 1), activation="sigmoid")(features)
    regression = keras.layers.Conv2D(9 * 4, (1, 1))(features)

    y_pred = keras.layers.concatenate([classification, regression])

    model = keras.models.Model(image, features)

    loss = keras_rcnn.losses.rpn.proposal(9, (224, 224), 16)

    def example(x):
        y_true, y_pred = x

        return loss(y_true, y_pred)

        # use the following for testing:
        # return y_true


    loss = keras.layers.Lambda(example)([y_true, y_pred])

    model = keras.models.Model([image, y_true], loss)

    model.add_loss(keras.backend.sum(loss, axis=None))

    model.compile("adam", loss=[None], loss_weights=[None])

    a = numpy.random.random((1, 224, 224, 3))
    b = numpy.random.random((1, 10, 4))

    model.fit([a, b], [None])
