_base_ = [
    '../_base_/models/faster_rcnn_swin_fpn.py',
    '../_base_/datasets/street_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

dataset_type = 'StreetDataset'
num_classes = 5
classes = ("car", "crosswalk", "light", "traffic_sign", "warning_line",)


#import os
data_root = '/home/xz/lzx/Swin-Transformer-Object-Detection/data/OmnidirectionalStreetViewDataset/equirectangular/'
# data_root = r"E:/ori_disks/D/fduStudy/labZXD/repos/datasets/OmnidirectionalStreetViewDataset/equirectangular/"
# python tools/train.py configs/swin/faster_rcnn_swin_tiny_patch4_window7_mstrain_480800_adamw_1x_streetwin.py
checkpoint_config = dict(interval=10)

model = dict(
    backbone=dict(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        ape=False,
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
    neck=dict(in_channels=[96, 192, 384, 768]))


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                 (736, 1333), (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True),
             ],
         ]),
    dict(type='BasketBallExpand',
         patches_y=7,
         align_type=['center', 'center2', 'left', 'right']),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(pipeline=train_pipeline,
               ann_file=data_root + 'train.json',
               img_prefix=data_root + 'JPEGImages/',
               classes=classes
               ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root + 'JPEGImages/',
        classes=classes
    ),
    test=dict(
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
lr_config = dict(step=[8, 11])
runner = dict(type='EpochBasedRunnerAmp', max_epochs=12)


# do not use mmdet version fp16
fp16 = None


# resume_from = "/home/hadoop/project/ZhixinLing/Swin-Transformer-Object-Detection/work_dirs/faster_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_street/latest.pth"
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=False,
)

