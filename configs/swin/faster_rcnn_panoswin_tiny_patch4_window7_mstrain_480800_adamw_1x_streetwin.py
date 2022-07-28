is_win32 = 1


_base_ = [
    '../_base_/models/faster_rcnn_panoswin_fpn.py',
    '../_base_/datasets/street_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

dataset_type = 'StreetDataset'
num_classes = 5
classes = ("car", "crosswalk", "light", "traffic_sign", "warning_line",)

patch_size = 4

#import os
data_root = 'data/OmnidirectionalStreetViewDataset/equirectangular/'
if is_win32:
    data_root = r"E:/ori_disks/D/fduStudy/labZXD/repos/datasets/OmnidirectionalStreetViewDataset/equirectangular/"
# python tools/train.py configs/swin/faster_rcnn_panoswin_tiny_patch4_window7_mstrain_480800_adamw_1x_streetwin.py
checkpoint_config = dict(interval=20)
# checkpoint_config = dict(interval=1) # WIN32 interval=1
embed_dim = 96 # WIN32 96
model = dict(
    backbone=dict(
        in_chans=3,
        embed_dim=embed_dim,
        depths=[2, 2, 2 + (1 - is_win32) * 4, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        patch_size=patch_size,
	    emb_conv_type='cnn',
        basketball_trans=False,
        ape=True,
        drop_path_rate=0.1,
        patch_norm=True,
        use_checkpoint=False
    ),
    roi_head=dict(
        bbox_head=
        dict(
            type='Shared2FCBBoxHead',
            # explicitly over-write all the `num_classes` field from default 80 to 5.
            num_classes=num_classes),
    ),
    neck=dict(in_channels=[embed_dim, 192, 384, 768]))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


train_resize_scales = [(480, 1333), (512, 1333), (544, 1333),
                                 (576, 1333),  (608, 1333), (640, 1333),
                                 (672, 1333), (704, 1333), (736, 1333),
                                 (768, 1333), (800, 1333)]
img_scales = [(400, 1333), (500, 1333), (600, 1333)]

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadPanoAnnotations', with_bbox=True, with_mask=False, bb_tangent2sphere=True),
    dict(type='PanoStretch', chance=1.0, kxy=(2.0,2.0)),
    dict(type='RollAug', chance=1.0, clip01=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=train_resize_scales,
                      multiscale_mode='value',
                      keep_ratio=True
                      )
             ],
             [
                 dict(type='Resize',
                      img_scale=img_scales,
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=train_resize_scales,
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='PanoCheck'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'pano_ratio_v']),
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
            dict(type='PanoCheck'),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'pano_ratio_v']),
        ])
]


data = dict(
    samples_per_gpu=4 - 3 * is_win32,
    workers_per_gpu=2 - 2 * is_win32,
    train=dict(pipeline=train_pipeline,
                type=dataset_type,
               ann_file=data_root + 'train.json',
               img_prefix=data_root + 'JPEGImages/',
               classes=classes
               ),
    val=dict(
        pipeline=test_pipeline,
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root + 'JPEGImages/',
        classes=classes
    ),
    test=dict(
        pipeline=test_pipeline,
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root + 'JPEGImages/',
        classes=classes
    ),
)

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

amplifier = 30
lr_config = dict(step=[8*amplifier, 11*amplifier])


if not is_win32:
    runner = dict(type='EpochBasedRunnerAmp', max_epochs=12 * amplifier)
    # do not use mmdet version fp16
    fp16 = None

    # resume_from = "/home/hadoop/project/ZhixinLing/Swin-Transformer-Object-Detection/work_dirs/faster_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_street/latest.pth"
    optimizer_config = dict(
        type="DistOptimizerHook",
        update_interval=1,
        grad_clip=None,
        coalesce=True,
        bucket_size_mb=-1,
        use_fp16=True,
    )
else:
    runner = dict(type='EpochBasedRunner', max_epochs=12 * 15)

