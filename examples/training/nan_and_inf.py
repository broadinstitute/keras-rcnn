# -*- coding: utf-8 -*-

"""
NaN (and Inf)
=============

A neural network whose layers or losses yield NaN or Inf values are a common
machine learning problem. They are especially obnoxious because it’s
difficult for experienced and inexperienced users alike to find the source (
or sources) of the problem. They usually originate from: a bug in Keras-RCNN
(or a Keras-RCNN dependency), numerical stability problems in a user’s
runtime environment (like CUDA or a related library), or, most commonly,
a fundamental problem in a model’s architecture or the underlying algorithms
(like an objective function). In the following example, I’ve summarized the
aforementioned issues alongside techniques to diagnose and resolve them.

Hyperparameters
---------------

Inappropriately configured hyperparameters are a common source of NaN and
Inf problems.

Learning rate
~~~~~~~~~~~~~

Inappropriate learning rates are a frequent source of NaN and Inf
frustration. A learning rate that’s too large will quickly, if not
immediately, yield NaN values.

.. code:: python

    model = keras_rcnn.models.RCNN(
        categories=categories.keys(),
        dense_units=512,
        input_shape=(224, 224, 3)
    )

    optimizer = keras.optimizers.SGD(0.1)

    model.compile(optimizer)

    model.fit_generator(
        epochs=1,
        generator=generator,
        validation_data=generator
    )

The easiest solution to remedy this problem is to use Keras’
LearningRateScheduler callback to enforce a schedule that steadily decrease
the learning rate:

.. code:: python

    import math

    def schedule(epoch_index):
        return 0.1 * numpy.power(0.5, numpy.floor((1 + epoch_index) / 1.0))

    learning_rates = [schedule(epoch_index) for epoch_index in range(0, 10)]

    matplotlib.pyplot.plot(learning_rates)

    model.fit_generator(
        callbacks = [
            keras.callbacks.LearningRateScheduler(schedule)
        ],
        epochs=10,
        generator=generator,
        validation_data=generator
    )

You could also consider writing a custom Keras callback that halves the
learning rate until your objective function yields reasonable values:

Penalties
~~~~~~~~~

Other hyperparameters may also play a role. For example, are your training
algorithms involve regularization terms? If so, are their corresponding
penalties set reasonably? Search a wider hyperparameter space with a few (
one or two) training epochs each to see if the NaNs could disappear.

Weight initialization
~~~~~~~~~~~~~~~~~~~~~

Some models can be very sensitive to the initialization of weight vectors.
If the weights are not properly initialized, then it is not surprising that
the model ends up with yielding NaNs.

Numerical stability
-------------------
"""


def main():
    pass


if __name__ == '__main__':
    main()
