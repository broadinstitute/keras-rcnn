import keras.layers
import keras_rcnn.models
import keras.layers
import keras.backend
import keras.models
import numpy
import keras_rcnn.losses.rpn
import keras_rcnn.models


def test_resnet50_rcnn():
    image    = keras.layers.Input(shape=(224, 224, 3), name='image')
    im_info  = keras.layers.Input(shape=(3,), name='im_info')
    gt_boxes = keras.layers.Input(shape=(None, 5,), name='gt_boxes')

    model = keras_rcnn.models.ResNet50RCNN([image, im_info, gt_boxes], 21, 300)

    model.compile(loss=None, optimizer="adam")


def test_rpn():
    options = {
        "activation": "relu",
        "kernel_size": (3, 3),
        "padding": "same"
    }

    image = keras.layers.Input(shape=(224, 224, 3), name='image')

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


    rpn_loss = keras_rcnn.losses.rpn.proposal(9, (224, 224), 16)

    def example(x, loss):
        y_true, y_pred = x
        loss_c, loss_r = loss(y_true, y_pred)
        return loss_c + loss_r

        # use the following for testing:
        # return y_true


    loss = keras.layers.Lambda(example, arguments={'loss': rpn_loss},)([y_true, y_pred])

    model = keras.models.Model([image, y_true], loss)

    model.add_loss(keras.backend.sum(loss, axis=None))

    model.compile("adam", loss=[None], loss_weights=[None])

    a = numpy.zeros((1, 224, 224, 3))

    y_true = numpy.array([
        [1, 1, 100, 100],
        [0, 0, 40, 50],
        [50, 99, 100, 203],
        [111, 5, 131, 34],
        [4, 60, 30, 90]])

    y_true = numpy.expand_dims(y_true, 0)

    model.fit([a, y_true], [None])
