Layers
######

.. contents:: Table of Contents
   :depth: 2

Anchor
======

The Anchor layer generates multi-aspect ratio and multiscale anchor bounding boxes with a corresponding category that classifies each anchor bounding box as an object or non-object.

Introduction
------------

At each sliding-window location, the RCNN model simultaneously predicts multiple region proposals, where the number of maximum possible proposals for each location is denoted :math:`k`. The regression convolutional layer has 4,000 outputs encoding the coordinates of  boxes, and the classification convolutional layer outputs 2,000 scores that estimate the probability of object or not object for each proposal. :math:`k` proposals are parameterized relative to  reference bounding boxes, which we call anchors. An anchor is centered at the sliding window in question, and is associated with a scale and aspect ratio.

(scale and aspect ratio figures)

By default, three scales and three aspect ratios are used, yielding :math:`k = 9` anchors at each sliding window position. For a convolutional feature map of size :math:`rc`, there are :math:`rck` anchors.

Translational symmetry
~~~~~~~~~~~~~~~~~~~~~~

The RCNN model is translation invariant in terms of the anchors and the functions that compute object proposals relative to the anchors. If one translates an object in an image, the proposal should translate and the same function should be able to predict the proposal in either location. This translation-invariant property is guaranteed.

The translation-invariant property also reduces the model size. The model has a (4 + 2) × 9-dimensional convolutional output layer in the case of k = 9 anchors.

Multi-Scale Anchors as Regression References
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model addresses multiple scales and aspect ratios by building a pyramid of anchors. The model classifies and regresses bounding boxes with reference to anchor boxes of multiple scales and aspect ratios. It only relies on images and feature maps of a single scale, and uses sliding windows on the feature map of a single size.

Because of this multiscale architecture, we can use the convolutional features computed on a single-scale image. The design of multiscale anchors is a key component for sharing features without extra cost for addressing scales.

Implementation
--------------

By default, the AnchorTarget layer uses three scales with bounding box areas of :math:`128^{2}`, :math:`256^{2}`, and :math:`512^{2}` pixels and three aspect ratios of 1:1, 1:2, and 2:1. The aspect ratios and scales hyperparameters were not carefully chosen for a particular dataset.

The model permits predictions larger than the underlying receptive field because such predictions are not impossible (one might still infer the extent of an object if only the middle of the object is visible).

The AnchorTarget layer ignores anchors that cross an image’s boundaries so they do not contribute to the loss.

For a typical 1,000 × 600 image, there will be roughly 20,000 (≈ 60 × 40 × 9) anchors in total. With the cross-boundary anchors ignored, there are about 6,000 anchors per image for training. If the boundary-crossing outliers are not ignored in training, they introduce large, difficult to correct error terms in the objective and training won’t converge. However, since the fully-convolutional region proposal network is applied to the entire image, cross-boundary object proposal bounding boxes might be generated but they are clipped to the image shape by the PropoalTarget layer.

.. autoclass:: keras_rcnn.layers.Anchor

ObjectDetection
===============

.. autoclass:: keras_rcnn.layers.ObjectDetection

ObjectProposal
==============

.. autoclass:: keras_rcnn.layers.ObjectProposal

ProposalTarget
==============

.. autoclass:: keras_rcnn.layers.ProposalTarget

RCNN
====

.. autoclass:: keras_rcnn.layers.RCNN

RegionOfInterest
================

.. autoclass:: keras_rcnn.layers.RegionOfInterest

RPN
===

.. autoclass:: keras_rcnn.layers.RPN

Upsample
========

.. autoclass:: keras_rcnn.layers.Upsample
