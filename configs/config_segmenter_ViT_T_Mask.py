import os

_base_ = [
    '../mmsegmentation/configs/_base_/models/segmenter_vit-b16_mask.py',
    '../mmsegmentation/configs/_base_/default_runtime.py'
]


load_from = os.getcwd().replace("\\", "/") + '/checkpoints/segmenter_vit-t_mask_8x1_512x512_160k_ade20k_20220105_151706-ffcf7509.pth'
device = 'cuda'

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_tiny_p16_384_20220308-cce8c795.pth'  # noqa

backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        embed_dims=192,
        num_heads=3,
    ),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=192,
        channels=192,
        num_classes=28,
        num_heads=3,
        embed_dims=192,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))


optimizer = dict(type='SGD', momentum=0.9, lr=0.001, weight_decay=0.0)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=12)
# runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=True, interval=1)
evaluation = dict(interval=1, metric='mIoU', pre_eval=True)

dataset_type = 'MyDataset'
data_root = os.getcwd().replace("\\", "/") + '/dataset'
work_dir = os.getcwd().replace("\\", "/") + '/weights'

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    train=dict(type=dataset_type,
               img_dir=data_root + '/img_dir/train',
               ann_dir=data_root + '/ann_dir/train',
               pipeline=train_pipeline),
    val=dict(type=dataset_type,
             img_dir=data_root + '/img_dir/val',
             ann_dir=data_root + '/ann_dir/val',
             pipeline=test_pipeline),
    test=dict(type=dataset_type,
              img_dir=data_root + '/img_dir/test',
              ann_dir=data_root + '/ann_dir/test',
              pipeline=test_pipeline),
    train_dataloader=dict(samples_per_gpu=4, workers_per_gpu=6, shuffle=True),
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=4, shuffle=False),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=4, shuffle=False))
