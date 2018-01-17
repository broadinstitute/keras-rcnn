Keras-RCNN
==========

.. image:: https://travis-ci.org/broadinstitute/keras-rcnn.svg?branch=master
    :target: https://travis-ci.org/broadinstitute/keras-rcnn

.. image::https://codecov.io/gh/broadinstitute/keras-rcnn/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/broadinstitute/keras-rcnn

keras-rcnn is *the* Keras package for region-based convolutional
neural networks.

Getting Started
---------------

.. code:: python

    import keras_rcnn.datasets
    import keras_rcnn.preprocessing

    training, validation, test = keras_rcnn.datasets.malaria.load_data()

    generator = keras_rcnn.preprocessing.ObjectDetectionGenerator()

    classes = {
        "rbc": 1,
        "not":2
    }

    generator = generator.flow(training, classes)

Create an RCNN instance:

.. code:: python

    import keras.layers
    import keras_rcnn.models

    image = keras.layers.input((None, None, 3))

    model = keras_rcnn.models.RCNN(image, classes=len(classes) + 1)

Specify your preferred optimizer and pass that to the compile method:

.. code:: python

    optimizer = keras.optimizers.Adam(0.001)

    model.compile(optimizer)

Train the model:

.. code:: python

    model.fit_generator(generator)

Finally, make a prediction from the trained model:

.. code:: python

    x = generator.next()[0]

    y_anchors, y_deltas, y_proposals, y_scores = model.predict(x)

Slack
-----

We’ve been meeting in the #keras-rcnn channel on the keras.io Slack
server. 

You can join the server by inviting yourself from the following website:

https://keras-slack-autojoin.herokuapp.com/
