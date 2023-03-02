# dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/1Tm2/chl/TT100K-COCO/'
#data_root = '/media/sjk/2Tm2/CHL/COCO/'
classes = ("i2",
                "i4",
                "i5",
                "il100",
                "il60",
                "il80",
                "io",
                "ip",
                "p10",
                "p11",
                "p12",
                "p19",
                "p23",
                "p26",
                "p27",
                "p3",
                "p5",
                "p6",
                "pg",
                "ph4",
                "ph4.5",
                "ph5",
                "pl100",
                "pl120",
                "pl20",
                "pl30",
                "pl40",
                "pl5",
                "pl50",
                "pl60",
                "pl70",
                "pl80",
                "pm20",
                "pm30",
                "pm55",
                "pn",
                "pne",
                "po",
                "pr40",
                "w13",
                "w32",
                "w55",
                "w57",
                "w59",
                "wo")


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        #type='RepeatDataset',
        #times=3,
        classes = classes,
        ann_file=data_root + 'annotations/instances_train2014.json',
        img_prefix=data_root + 'train2014/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes = classes,
        ann_file=data_root + 'annotations/instances_val2014.json',
        img_prefix=data_root + 'val2014/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes = classes,
        ann_file=data_root + 'annotations/instances_val2014.json',
        img_prefix=data_root + 'val2014/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')