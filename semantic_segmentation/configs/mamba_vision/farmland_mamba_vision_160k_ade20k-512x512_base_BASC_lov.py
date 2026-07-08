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
        pretrained="/mnt/c/seg_dir/mambavision_main/semantic_segmentation/ckpts/mambavision_base_21k.pth.tar",
        depths = (3, 3, 10, 5),
        num_heads = (2, 4, 8, 16),
        window_size = (8, 8, 64, 32),
        dim = 128,
        in_dim = 64,
        mlp_ratio = 4,
        drop_path_rate = 0.4,
        norm_layer="ln2d",
        layer_scale = 1e-5,
        ),
    decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=2),
                    #  loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                    #               dict(type='LovaszLoss',reduction='none',per_image=True,loss_weight=0.3)]),          

    auxiliary_head=dict(in_channels=512, num_classes=2
                        # loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
                        #              dict(type='LovaszLoss',reduction='none',per_image=True,loss_weight=0.1)])
                        ))

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00003, betas=(0.9, 0.999), weight_decay=0.01),
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

# This model is trained on 2 nodes, 16 GPUs, 1 image per GPU
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=2)
test_dataloader = val_dataloader




# 指令
# $env:CUDA_VISIBLE_DEVICES=1
# CUDA_VISIBLE_DEVICES=2 python semantic_segmentation/tools/train.py --config semantic_segmentation/configs/mamba_vision/farmland_mamba_vision_160k_ade20k-512x512_small_BASC_lov.py  --work-dir ./work_dir/farmland_mamba_small_BASC_lov  --amp 


# CUDA_VISIBLE_DEVICES=2 python semantic_segmentation/tools/train.py --config semantic_segmentation/configs/mamba_vision/farmland_mamba_vision_160k_ade20k-512x512_base_BASC_lov.py  --work-dir ./work_dir/farmland_mamba_base_BASC_lov  --amp 
