import keras.backend
import keras.utils
import numpy

import keras_rcnn.backend
import keras_rcnn.layers


class TestProposalTarget:

    def test_find_foreground_and_background_proposal_indices(self):
        #1
        overlaps = numpy.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.4, 0.0, 0.1, 0.2],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.2, 0.1, 0.6, 0.7],
        [0.0, 0.0, 0.9, 0.0],
        [0.01, 0.1, 0.2, 0.3],
        [0.0, 0.0, 0.5, 1.0],
        [0.3, 0.0, 0.0, 1.0]
        ])
        max_overlaps = keras.backend.max(overlaps, axis=1)
        fg_fraction = 0.5
        batchsize = 3
        p = keras_rcnn.layers.ProposalTarget()
        keep_inds = p.find_foreground_and_background_proposal_indices(max_overlaps)
        keep_inds = keras.backend.eval(keep_inds)
        expected_fg = numpy.array([2, 3, 4, 5, 7, 8])

        for i in range(int(numpy.round(fg_fraction * batchsize))):
            assert keep_inds[i] in expected_fg
        for i in range(int(numpy.round(fg_fraction * batchsize)), batchsize):
            assert keep_inds[i] not in expected_fg

        #2
        batchsize = 8
        p = keras_rcnn.layers.ProposalTarget()
        keep_inds = p.find_foreground_and_background_proposal_indices(max_overlaps)
        keep_inds = keras.backend.eval(keep_inds)
        expected_fg = numpy.array([2, 3, 4, 5, 7, 8])
        expected_bg = numpy.array([1, 6])

        fg_rois = numpy.minimum(int(numpy.round(fg_fraction * batchsize)), len(expected_fg))
        bg_rois = numpy.minimum(batchsize - fg_rois, len(expected_bg))
        for i in range(fg_rois):
            assert keep_inds[i] in expected_fg
        for i in range(fg_rois, fg_rois + bg_rois):
            assert keep_inds[i] in expected_bg

        #3
        batchsize = 16
        p = keras_rcnn.layers.ProposalTarget(fg_fraction=fg_fraction,
                                             fg_thresh=0.5,
                                             bg_thresh_hi=0.5,
                                             bg_thresh_lo=0.1,
                                             batchsize=batchsize,
                                             num_images=1)
        keep_inds = p.get_fg_bg_rois(max_overlaps)
        keep_inds = keras.backend.eval(keep_inds)
        expected_fg = numpy.array([2, 3, 4, 5, 7, 8])
        expected_bg = numpy.array([1, 6])

        fg_rois = numpy.minimum(int(numpy.round(fg_fraction * batchsize)), len(expected_fg))
        bg_rois = numpy.minimum(batchsize - fg_rois, len(expected_bg))
        for i in range(fg_rois):
            assert keep_inds[i] in expected_fg
        for i in range(fg_rois, fg_rois + bg_rois):
            assert keep_inds[i] in expected_bg

        #4
        overlaps = numpy.array([
        [0.2, 0.0, 0.0, 0.0],
        [0.4, 0.0, 0.1, 0.2],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.2, 0.1, 0.6, 0.7],
        [0.0, 0.0, 0.9, 0.0],
        [0.01, 0.1, 0.2, 0.3],
        [0.0, 0.0, 0.5, 1.0],
        [0.3, 0.0, 0.0, 1.0]
        ])
        max_overlaps = keras.backend.max(overlaps, axis=1)
        batchsize = 6
        p = keras_rcnn.layers.ProposalTarget(fg_fraction=fg_fraction,
                                             fg_thresh=0.5,
                                             bg_thresh_hi=0.5,
                                             bg_thresh_lo=0.1,
                                             batchsize=batchsize,
                                             num_images=1)
        keep_inds = p.get_fg_bg_rois(max_overlaps)
        keep_inds = keras.backend.eval(keep_inds)
        expected_fg = numpy.array([2, 3, 4, 5, 7, 8])
        expected_bg = numpy.array([0, 1, 6])

        fg_rois = numpy.minimum(int(numpy.round(fg_fraction * batchsize)), len(expected_fg))
        bg_rois = numpy.minimum(batchsize - fg_rois, len(expected_bg))
        for i in range(fg_rois):
            assert keep_inds[i] in expected_fg
        for i in range(fg_rois, fg_rois + bg_rois):
            assert keep_inds[i] in expected_bg
