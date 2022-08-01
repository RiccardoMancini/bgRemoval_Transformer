import os

_base_ = [
    '../mmsegmentation/configs/_base_/models/segmenter_vit-b16_mask.py',
    '../mmsegmentation/configs/_base_/default_runtime.py'
]

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_base_p16_384_20220308-96dfe169.pth'  # noqa
load_from = os.getcwd().replace("\\", "/") + '/checkpoints/segmenter_vit' \
                                            '-b_mask_8x1_512x512_160k_ade20k_20220105_151706-bc533b08.pth'
device = 'cuda'

backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
model = dict(
    type='EncoderDecoder',
    init_cfg=checkpoint,
    backbone=dict(
        type='VisionTransformer',
        img_size=(512, 512),
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        final_norm=True,
        norm_cfg=backbone_norm_cfg,
        with_cls_token=True,
        interpolate_mode='bicubic',
    ),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=768,
        channels=768,
        num_classes=28,
        num_layers=2,
        num_heads=12,
        embed_dims=768,
        dropout_ratio=0.0,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(480, 480)),
)

optimizer = dict(type='SGD', momentum=0.9, lr=0.001, weight_decay=0.0)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=10)
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
    train_dataloader=dict(samples_per_gpu=2, workers_per_gpu=6, shuffle=True),
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=4, shuffle=False),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=6, shuffle=False))
