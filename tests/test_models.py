import keras.layers
import keras_rcnn.models
import keras.layers
import keras.backend
import keras.models
import numpy
import keras_rcnn.models


def test_resnet50_rcnn():
    image    = keras.layers.Input(shape=(224, 224, 3), name='image')
    im_info  = keras.layers.Input(shape=(3,), name='im_info')
    gt_boxes = keras.layers.Input(shape=(None, 5,), name='gt_boxes')

    model = keras_rcnn.models.ResNet50RCNN([image, im_info, gt_boxes], 21, 300)

    model.compile(loss=None, optimizer="adam")
