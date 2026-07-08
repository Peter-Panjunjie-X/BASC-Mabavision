_base_ = [
    '../_base_/models/upernet_swin.py', 
    '../_base_/datasets/farmland.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)


model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MM_mamba_vision_BASC',
        out_indices=(0, 1, 2, 3),
        pretrained="/mnt/c/seg_dir/mambavision_main/semantic_segmentation/ckpts/mambavision_small_1k.pth.tar",
        depths = (3, 3, 7, 5),
        num_heads = (2, 4, 8, 16),
        window_size = (8, 8, 160, 56),
        dim = 96,
        in_dim = 64,
        mlp_ratio = 4,
        # 
        # drop_path_rate = 0.7,
        drop_path_rate=0.5,

        norm_layer="ln2d",
        layer_scale = None
        ),
    # decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=2),
    # auxiliary_head=dict(in_channels=384, num_classes=2))
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        align_corners=False,
        loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                     dict(type='LovaszLoss',reduction='none',per_image=True,loss_weight=0.3)]),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        align_corners=False,
        loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
                     dict(type='LovaszLoss',reduction='none',per_image=True,loss_weight=0.1)])
        )

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00007, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]


# This model is trained on 1 node, 8 GPUs, 2 image per GPU
train_dataloader = dict(batch_size=3)
val_dataloader = dict(batch_size=3)
test_dataloader = val_dataloader



# 指令
# $env:CUDA_VISIBLE_DEVICES=1
# CUDA_VISIBLE_DEVICES=3 python semantic_segmentation/tools/train.py --config semantic_segmentation/configs/mamba_vision/farmland_mamba_vision_160k_ade20k-512x512_small_BASC_lov.py  --work-dir ./work_dir/farmland_mamba_small_BASC_lov  --amp 

