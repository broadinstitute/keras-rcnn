import keras.backend
import keras.utils.test_utils
import numpy

import keras_rcnn.backend
import keras_rcnn.layers


def test_get_config():
    layer = keras_rcnn.layers.ProposalTarget()

    expected = {
        "background_threshold": (0.1, 0.5),
        "foreground": 0.5,
        "foreground_threshold": (0.5, 1.0),
        "maximum_proposals": 32,
        "name": 'proposal_target_1',
        "trainable": True
    }

    assert layer.get_config() == expected


def test_proposal_target():
    keras.backend.set_learning_phase(True)

    a = keras.layers.Input((None, 4))
    b = keras.layers.Input((None, 2))
    c = keras.layers.Input((None, 4))

    layer = keras_rcnn.layers.ProposalTarget()

    x, y, z = layer([a, b, c])

    model = keras.models.Model([a, b, c], [x, y, z])

    assert model.output_shape == [(1, None, 8), (1, None, 2), (1, None, 4)]

    a = numpy.random.random((1, 10, 4))
    b = numpy.random.random((1, 10, 2))
    c = numpy.random.random((1, 300, 4))

    x, y, z = model.predict([a, b, c])

    assert x.shape == (1, 32, 8), x.shape
    assert y.shape == (1, 32, 2), y.shape
    assert z.shape == (1, 32, 4), z.shape


def test_sample():
    proposal_target = keras_rcnn.layers.ProposalTarget(maximum_proposals=8)
    all_rois = numpy.array([
        [  50.71208954,   70.44628143,   90.61702728,   99.66291046],
        [  55.45672607,   67.94913483,   86.19535828,   98.13130951],
        [ 107.53626251,    1.19126511,  127.        ,   40.5984726 ],
        [  50.50813293,   79.33403778,   87.0002594 ,  117.19319916],
        [ 101.94552612,    0.        ,  125.88674927,   37.46565247],
        [   0.        ,    0.        ,  127.        ,   88.        ],
        [   0.        ,   24.        ,  127.        ,  120.        ],
        [  98.8294754 ,   21.01197243,  127.        ,   59.66217804],
        [  94.43806458,    3.28402901,  127.        ,   44.11544037],
        [   0.        ,   56.        ,  127.        ,  127.        ],
        [   0.        ,    0.        ,  127.        ,   56.        ],
        [  28.        ,   40.        ,  127.        ,  127.        ],
        [  91.25584412,    0.        ,  116.2077179 ,   32.36594391],
        [   0.        ,   40.        ,  100.        ,  127.        ],
        [   0.        ,    8.        ,  100.        ,  104.        ],
        [  16.        ,    0.        ,  127.        ,  127.        ],
        [  28.        ,    8.        ,  127.        ,  104.        ],
        [   0.        ,    0.        ,  100.        ,   72.        ],
        [  50.74036407,   94.12261963,   86.55892181,  127.        ],
        [   0.        ,   72.        ,  100.        ,  127.        ],
        [ 103.20545959,   86.37854004,  127.        ,  127.        ],
        [  40.        ,    0.        ,  127.        ,   88.        ],
        [  56.        ,    0.        ,  127.        ,   72.        ],
        [  28.        ,   72.        ,  127.        ,  127.        ],
        [  56.        ,   56.        ,  127.        ,  127.        ],
        [  76.        ,   32.        ,  127.        ,  127.        ],
        [   0.        ,   56.        ,   72.        ,  127.        ],
        [   0.        ,    0.        ,   72.        ,   88.        ],
        [  76.        ,    0.        ,  127.        ,   96.        ],
        [   0.        ,   32.        ,   52.        ,  127.        ],
        [  56.        ,    0.        ,  127.        ,  120.        ],
        [   0.        ,    0.        ,   52.        ,   96.        ],
        [   0.        ,   24.        ,   72.        ,  127.        ],
        [  97.16458893,   98.0249939 ,  127.        ,  127.        ],
        [ 108.26307678,   18.04890442,  127.        ,   51.71542358],
        [  94.32286072,    0.        ,  127.        ,   22.97975922],
        [   7.79359531,    0.        ,   36.87153625,   17.60238075],
        [   0.        ,    0.        ,   84.        ,  127.        ],
        [   0.        ,  106.34275055,   49.92878723,  127.        ],
        [   0.        ,   99.0591507 ,   35.27072525,  127.        ],
        [  14.40047836,  102.94164276,   66.9264679 ,  127.        ],
        [  19.59460831,   94.52481079,   58.59706497,  127.        ],
        [   0.        ,    0.        ,   41.47511673,   30.94105721],
        [   0.        ,   91.5510025 ,   23.51871109,  127.        ],
        [  98.41709137,   32.52594757,  127.        ,   70.71497345],
        [  83.86433411,    0.        ,  126.68435669,   17.05216599],
        [   4.25742149,  103.77056885,   40.9630661 ,  127.        ],
        [  92.50645447,   18.36931801,  127.        ,   74.69480896],
        [   0.        ,    0.        ,   23.31191254,   45.55099106],
        [   0.        ,   85.20574951,   18.08465958,  127.        ],
        [  34.76586914,   91.0597229 ,   70.21446991,  127.        ],
        [  31.57422256,   98.00201416,   82.33876038,  127.        ],
        [   0.        ,    1.70124626,   18.00316238,   58.02664948],
        [  47.77222824,  100.66429138,   97.60723877,  127.        ],
        [  49.00650024,    0.        ,   87.90568542,   18.85537338],
        [   2.66373634,    0.        ,   31.10864067,   32.25665665],
        [   7.98152542,   98.91853333,   75.25076294,  127.        ],
        [  96.10111237,   82.96366882,  127.        ,  124.20684814],
        [ 100.59230042,   51.73586655,  127.        ,   90.42410278],
        [  62.81672668,  105.77951813,  100.79664612,  127.        ],
        [  39.32888031,   78.95227814,   70.62407684,  127.        ],
        [  83.66664124,    0.        ,  123.00033569,   42.14031601],
        [  35.41271973,   96.42881775,  102.99723816,  127.        ],
        [  86.72212219,  104.05187988,  121.45080566,  127.        ],
        [  34.50315857,    0.        ,   72.09360504,   21.45676422],
        [  49.67893219,   85.49975586,   94.36956024,  122.12127686],
        [  16.8948822 ,    0.        ,   57.12744904,   23.87054443],
        [   0.        ,    0.        ,   49.20021057,   22.89399529],
        [  92.50393677,   31.71694946,  116.82061768,   75.54193115],
        [ 101.27336884,   68.32675171,  127.        ,  107.76806641],
        [ 103.63972473,   64.72322845,  127.        ,  121.69217682],
        [   0.        ,    2.98220253,   31.56446838,   48.02368927],
        [  98.78588104,   41.22549438,  127.        ,   98.6265564 ],
        [  43.14202118,   83.98725128,   81.71795654,  126.31203461],
        [  77.02322388,   50.71312714,  116.16117859,   85.06006622],
        [   0.        ,   81.65694427,   32.60420227,  123.96698761],
        [ 103.44710541,   14.21045494,  127.        ,   40.28822327],
        [  69.66470337,    0.        ,  110.63270569,   21.06321716],
        [   0.        ,    3.10115814,   19.54813766,   39.10132217],
        [   0.        ,   18.57296562,   24.90319443,   61.45012665],
        [  80.68254089,  100.47480774,  127.        ,  127.        ],
        [  22.4812355 ,    0.        ,   51.37631607,   19.1945343 ],
        [  57.52688599,   90.20191956,   91.05926514,  127.        ],
        [  95.99333191,   28.11805344,  127.        ,   54.7819252 ],
        [  66.12437439,  109.61106873,  112.16593933,  127.        ],
        [   0.        ,   34.92645645,   23.40029144,   76.49604797],
        [   0.        ,   24.81159973,   30.09602737,   64.4962616 ],
        [  71.15076447,    0.        ,   98.23110199,   24.22634506],
        [  57.86538696,   69.31452942,  100.78405762,  105.31706238],
        [  56.96481705,    0.        ,   84.40791321,   16.61228561],
        [  52.54270172,   71.56542969,   84.0104599 ,  106.01870728],
        [  59.92431641,   51.18797302,   93.5614624 ,   96.81411743],
        [  46.53039551,   60.00773621,   76.70639038,  105.10090637],
        [   0.        ,   51.24388123,   20.01187706,   91.44229126],
        [   0.        ,   40.91738892,   29.76662064,   79.57460785],
        [   0.        ,   66.71936798,   23.33432007,  107.1989212 ],
        [  52.53593826,   56.81362534,   87.6943512 ,   80.978302  ],
        [  23.93180656,    0.        ,   65.24099731,   21.62162399],
        [  52.88409424,   62.21540833,   86.62045288,   94.69174194],
        [  62.93517303,    0.        ,   97.4132309 ,   21.67671776],
        [  80.08380127,   71.31486511,  112.64286804,  106.20582581],
        [   0.        ,   75.59233093,   29.57452393,  115.32017517],
        [   0.        ,   94.73368835,   29.12003326,  122.7570343 ],
        [  56.74416351,   62.87185669,   83.51163483,  102.64590454],
        [  56.4704361 ,   67.04311371,   88.26943207,  107.08548737],
        [  61.38975525,   99.27015686,  116.5189209 ,  127.        ],
        [  74.17579651,    0.        ,  111.2371521 ,   29.42921066],
        [  89.39240265,   47.07178497,  126.46971893,   87.31560516],
        [  69.44123077,   47.38218307,  108.40250397,   79.6802063 ],
        [   0.        ,   57.49836731,   29.66657829,   95.89842224],
        [  78.95790863,   37.56030273,  115.78124237,   78.61038971],
        [   1.05501747,   49.41056061,   23.29784775,   78.91101837],
        [  85.10821533,   96.18637848,  113.51568604,  127.        ],
        [   0.        ,    4.97987747,   42.77118683,   39.50862122],
        [  20.71508408,   86.12203979,   66.83483887,  127.        ],
        [  92.37216949,    5.15529823,  120.51592255,   36.1371994 ],
        [  18.54267502,    7.85336876,   47.64195633,   40.28590393],
        [  70.94502258,   10.51726341,   96.90571594,   49.78765106],
        [  78.40616608,   34.65525818,  110.38196564,   65.63249969],
        [  62.15895844,   59.15465546,  105.94535065,   89.95876312],
        [   1.27451801,   69.83500671,   25.79981232,   97.2482605 ],
        [  62.73934174,   83.32158661,  101.73313141,  120.8647995 ],
        [  45.75923157,   59.99327469,   71.63579559,   98.12197876],
        [  56.19100189,   14.82108879,   84.26691437,   44.36116791],
        [  65.84653473,   42.47505951,   97.59632111,   82.20253754],
        [  69.37020874,   94.50706482,  108.35041809,  127.        ],
        [  37.10490417,    9.61324596,   64.72504425,   38.63171387],
        [  85.3844986 ,   66.10962677,  110.31502533,  107.09311676],
        [  37.36283112,   60.59078217,   61.06542206,   91.03170013],
        [  67.89260864,   71.75512695,  106.5725708 ,  102.57484436],
        [  77.99923706,   16.48104858,  107.95568848,   48.56900024],
        [  89.20878601,   71.41056824,  125.7568512 ,  107.29753113],
        [   5.91212463,   32.58755112,   41.92187881,   62.24328232],
        [  52.27661133,   39.26787949,   81.46669006,   69.80540466],
        [  47.65753937,   40.49248505,   76.20034027,   76.36449432],
        [  57.34262848,    7.1027317 ,   82.6523056 ,   36.4445343 ],
        [  18.88934326,   92.23389435,   47.76045227,  124.64731598],
        [   9.30751991,   49.58087158,   42.02339172,   77.00914764],
        [  45.11285782,   61.83740234,   75.83686829,   87.559021  ],
        [  15.24866962,   45.1537323 ,   40.4258461 ,   72.44976044],
        [  26.69694901,   13.21245193,   60.47519302,   42.36695862],
        [   3.4413681 ,   88.63192749,   40.75950241,  116.03289795],
        [  33.86211777,   72.85886383,   64.36898804,  111.14054108],
        [  27.03687096,   89.1621933 ,   60.37636566,  116.18755341],
        [  32.48144913,   47.85022736,   59.34322739,   74.05560303],
        [  15.24499512,   57.00760651,   38.60559464,   84.32082367],
        [   9.1018734 ,   68.70444489,   41.70313263,   95.08370209],
        [  29.1010685 ,   40.56064987,   54.72600555,   74.27661133],
        [  63.39616394,   20.40607834,   91.86817932,   48.19306564],
        [  59.84752655,   49.76169586,   92.02407074,   74.77848816],
        [  22.08538437,   24.85385132,   47.61610031,   57.60162354],
        [  46.91257477,   17.56220245,   76.49073029,   44.3457756 ],
        [  31.83808899,   58.19957733,   59.74316406,   81.0170517 ],
        [  32.10147095,   74.45246124,   59.8211441 ,   97.25855255],
        [  56.94638062,   24.92946815,   84.93023682,   58.81396484],
        [  17.57852745,   71.32565308,   40.61666107,  100.14833069],
        [  44.46640396,   25.70672989,   69.00811768,   59.31702042],
        [  31.93849182,   42.83921432,   60.13134766,   66.11750793],
        [  47.41349792,   45.91717911,   75.39443207,   69.51273346]])
    all_rois = keras.backend.cast(all_rois, keras.backend.floatx())
    gt_boxes = numpy.array([[ 105.,    7.,  127.,   44.],
                            [  51.,   78.,   92.,   98.]])
    gt_boxes = keras.backend.cast(gt_boxes, keras.backend.floatx())
    gt_labels = numpy.array([[0, 1, 0], [0, 0, 1]])
    gt_labels = keras.backend.cast(gt_labels, keras.backend.floatx())

    rois, labels, bbox_targets = proposal_target.sample(all_rois, gt_boxes, gt_labels)


    out = keras.backend.concatenate([rois, labels, bbox_targets], axis=1)
    out = keras.backend.eval(out)
    rois = out[:, :4]
    labels = out[:, 4:(4+3)]
    bbox_targets = out[:, (4+3):]

    gt_boxes = keras.backend.eval(gt_boxes)

    pred = keras_rcnn.backend.bbox_transform_inv(rois, bbox_targets)
    pred = keras.backend.eval(pred)

    classes = numpy.argmax(labels, -1)

    label = classes[0]
    assert numpy.array_equal(pred[0][4*label : 4*label+4], gt_boxes[0]) or numpy.array_equal(pred[0][4*label : 4*label+4], gt_boxes[1])

    label = classes[1]
    assert numpy.array_equal(pred[1][4*label : 4*label+4], gt_boxes[0]) or numpy.array_equal(pred[1][4*label : 4*label+4], gt_boxes[1])

    assert labels.shape == (8, 3)
    assert labels[4:, 0].sum() == 4
    assert labels[:4, 1:].sum() == 4
    assert bbox_targets[4:].sum() == 0
    assert bbox_targets[:4, :4].sum() == 0


def test_set_label_background():
    p = keras_rcnn.layers.ProposalTarget()

    gt_labels = numpy.array([1, 2, 3, 1,
                  1, 2, 3, 3,
                  3, 1, 2, 2,
                  1, 2, 2, 2,
                  2, 1, 3, 3,
                  1, 1, 3, 1,
                  2, 2, 3, 2,
                  3, 3, 1, 1])
    gt_labels = keras.backend.one_hot(gt_labels, 4)
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
    gt_assignment = keras.backend.argmax(overlaps, axis=1)
    keep_inds = p.find_foreground_and_background_proposal_indices(max_overlaps)
    all_labels = keras.backend.gather(gt_labels, gt_assignment)
    labels = keras.backend.gather(all_labels, keep_inds)

    result = p.set_label_background(labels)
    result = keras.backend.eval(result)
    assert result[-1].sum() == 0
    assert result[0].sum() == 1
    assert result[1].sum() == 1


def test_get_bbox_regression_labels():
    p = keras_rcnn.layers.ProposalTarget()

    bbox_target_data = numpy.array([
        [0, 0, 0, 0],
        [.1, .2, -.1, -.3],
        [1, 2, 3, 4],
        [.1, .2, -.1, -.3],
        [1, 2, 3, 4]
    ])
    num_classes = 3
    labels = numpy.array([[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]])
    labels = numpy.argmax(labels, axis=1)
    bbox_targets = p.get_bbox_regression_labels(bbox_target_data, labels, num_classes)
    bbox_targets = keras.backend.eval(bbox_targets)

    assert bbox_targets.shape == (5, 4 * num_classes)
    expected = numpy.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, .1, .2, -.1, -.3],
        [0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0]
    ])
    numpy.testing.assert_array_almost_equal(bbox_targets, expected)


def test_get_bbox_targets():
    p = keras_rcnn.layers.ProposalTarget()

    rois = numpy.array([[0, 14.4, 30.1, 99.], [40., 3.2, 55.5, 33.7], [30.7, 1.2, 66.5, 23.8], [50.7, 50.2, 86.5, 63.8]])
    gt_boxes = numpy.array([[3., 20., 22., 100.], [40., 3., 55., 33.], [30., 1., 66., 24.], [15., 18., 59., 59.]])
    num_classes = 4
    labels = numpy.array([1, 1, 2, 1])

    bbox_targets = p.get_bbox_targets(rois, gt_boxes, labels, num_classes)
    bbox_targets = keras.backend.eval(bbox_targets)
    expected = numpy.array([
        [ 0.        ,  0.        ,  0.        ,  0.        ,
          -0.08471760797342194, 0.039007092198581526, -0.46008619258838973, -0.05590763193829595,
          0.        ,  0.        , 0.        ,  0.        ,
          0.        ,  0.        ,  0.        , 0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ,
          -0.016129032258064516, -0.014754098360655828, -0.03278982282299084, -0.016529301951210697,
          0.        ,  0.        , 0.        ,  0.        ,
          0.        ,  0.        ,  0.        , 0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ,
          0.        , 0.        ,  0.        ,  0.        ,
          -0.01675977653631289, 0.0, 0.0055710450494554295, 0.017544309650909525,
          0.        ,  0.        ,  0.        , 0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ,
          -0.88268156424581, -1.3602941176470593, 0.20624174051160657, 1.103502273962302,
          0.        ,  0.        ,  0.        ,  0.        ,
          0.        ,  0.        ,  0.        ,  0.        ]])
    numpy.testing.assert_array_almost_equal(bbox_targets, expected)


def test_find_foreground_and_background_proposal_indices():
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
    p = keras_rcnn.layers.ProposalTarget(maximum_proposals=batchsize)
    keep_inds = p.find_foreground_and_background_proposal_indices(max_overlaps)
    keep_inds = keras.backend.eval(keep_inds)
    expected_fg = numpy.array([2, 3, 4, 5, 7, 8])
    expected_bg = numpy.array([1, 6])

    assert keep_inds[0] in expected_fg, keep_inds
    assert keep_inds[1] in expected_bg, keep_inds
    assert keep_inds[2] in expected_bg, keep_inds
    assert len(keep_inds) == 3

    #2
    batchsize = 8
    p = keras_rcnn.layers.ProposalTarget(maximum_proposals=batchsize)
    keep_inds = p.find_foreground_and_background_proposal_indices(max_overlaps)
    keep_inds = keras.backend.eval(keep_inds)
    expected_fg = numpy.array([2, 3, 4, 5, 7, 8])
    expected_bg = numpy.array([1, 6])

    fg_rois = numpy.minimum(int(numpy.round(fg_fraction * batchsize)), len(expected_fg))
    bg_rois = numpy.minimum(batchsize - fg_rois, len(expected_bg))
    assert keep_inds[0] in expected_fg
    assert keep_inds[1] in expected_fg
    assert keep_inds[2] in expected_fg
    assert keep_inds[3] in expected_fg
    assert keep_inds[4] in expected_bg
    assert keep_inds[5] in expected_bg
    assert len(keep_inds) == 6

    #3
    batchsize = 16
    p = keras_rcnn.layers.ProposalTarget(maximum_proposals=batchsize)
    keep_inds = p.find_foreground_and_background_proposal_indices(max_overlaps)
    keep_inds = keras.backend.eval(keep_inds)
    expected_fg = numpy.array([2, 3, 4, 5, 7, 8])
    expected_bg = numpy.array([1, 6])

    fg_rois = numpy.minimum(int(numpy.round(fg_fraction * batchsize)), len(expected_fg))
    bg_rois = numpy.minimum(batchsize - fg_rois, len(expected_bg))
    assert keep_inds[0] in expected_fg
    assert keep_inds[1] in expected_fg
    assert keep_inds[2] in expected_fg
    assert keep_inds[3] in expected_fg
    assert keep_inds[4] in expected_fg
    assert keep_inds[5] in expected_fg
    assert keep_inds[6] in expected_bg
    assert keep_inds[7] in expected_bg
    assert len(keep_inds) == 8

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
    p = keras_rcnn.layers.ProposalTarget(maximum_proposals=batchsize)
    keep_inds = p.find_foreground_and_background_proposal_indices(max_overlaps)
    keep_inds = keras.backend.eval(keep_inds)
    expected_fg = numpy.array([2, 3, 4, 5, 7, 8])
    expected_bg = numpy.array([0, 1, 6])

    fg_rois = numpy.minimum(int(numpy.round(fg_fraction * batchsize)), len(expected_fg))
    bg_rois = numpy.minimum(batchsize - fg_rois, len(expected_bg))
    assert keep_inds[0] in expected_fg
    assert keep_inds[1] in expected_fg
    assert keep_inds[2] in expected_fg
    assert keep_inds[3] in expected_bg
    assert keep_inds[4] in expected_bg
    assert keep_inds[5] in expected_bg
    assert len(keep_inds) == 6

def test_sample_indices():
    p = keras_rcnn.layers.ProposalTarget()
    indices = numpy.array([0, 3, 4, 10, 22])
    sampled = p.sample_indices(indices, 3)
    sampled = keras.backend.eval(sampled)
    assert sampled.shape == (3,)
    for i in sampled:
        assert i in indices

    sampled = p.sample_indices(indices, 4)
    sampled = keras.backend.eval(sampled)
    assert sampled.shape == (4,)
    for i in sampled:
        assert i in indices
