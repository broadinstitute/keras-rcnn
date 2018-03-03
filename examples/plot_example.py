# -*- coding: utf-8 -*-

"""
Plotting the exponential function
=================================

A simple example for ploting two figures of a exponential
function in order to test the autonomy of the gallery
stacking multiple images.
"""

import numpy
import matplotlib.pyplot


def main():
    x = numpy.linspace(-1, 2, 100)
    y = numpy.exp(x)

    matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(x, y)
    matplotlib.pyplot.xlabel('$x$')
    matplotlib.pyplot.ylabel('$\exp(x)$')

    matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(x, -numpy.exp(-x))
    matplotlib.pyplot.xlabel('$x$')
    matplotlib.pyplot.ylabel('$-\exp(-x)$')

    matplotlib.pyplot.show()


if __name__ == '__main__':
    main()
