_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 24  # bs: total bs in all gpus
mix_prob = 0
empty_cache = False
enable_amp = True
find_unused_parameters = True

# trainer
train = dict(
    type="MultiDatasetTrainer",
)


# model settings
model = dict(
    type="all-level-v0",
    # num_classes=16,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v6m5",
        in_channels=4,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(256, 256, 256, 256, 256),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(256, 256, 256, 256),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        cls_mode=False,
        pdnorm_bn=True,
        pdnorm_ln=True,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo"),
    ),
    ins_head=dict(
        type="Ins-Head",
    ),
    # fmt: off
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    # backbone_out_channels=96,
    context_channels=256,
    conditions=("nuScenes", "SemanticKITTI", "Waymo"),
    num_classes=(16, 19, 22) 
    # fmt: on
)
# scheduler settings
epoch = 50
eval_epoch = 50
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
param_dicts = [dict(keyword="block", lr=0.0002)]

# dataset settings
# dataset_type = "NuScenesDataset"
# data_root = "data/nuscenes"
ignore_index = -1
names=['CAR', 'TRUCK', 'BUS', 'OTHER_VEHICLE', 'MOTORCYCLIST', 'BICYCLIST', 'PEDESTRIAN', 'SIGN',
              'TRAFFIC_LIGHT', 'POLE', 'CONSTRUCTION_CONE', 'BICYCLE', 'MOTORCYCLE', 'BUILDING', 'VEGETATION',
              'TREE_TRUNK', 'CURB', 'ROAD', 'LANE_MARKER', 'OTHER_GROUND', 'WALKABLE', 'SIDEWALK']


data = dict(
    num_classes=22,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type="ConcatDataset",
        datasets=[
            # nuScenes
            dict(
                type="NuScenesDataset",
                split="train",
                data_root="/data/sets/nuScenes",
                transform=[
                    # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
                    # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis='z', p=0.75),
                    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
                    # dict(type="RandomRotate", angle=[-1/6, 1/6], axis='x', p=0.5),
                    # dict(type="RandomRotate", angle=[-1/6, 1/6], axis='y', p=0.5),
                    # dict(type="PointClip", point_cloud_range=(-50, -50, -5, 50, 50, 3)),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                    dict(type="GridSample", grid_size=0.1, hash_type="fnv", mode="train",
                         keys=("coord", "strength", "segment"), return_grid_coord=True),
                    # dict(type="SphereCrop", point_max=1000000, mode="random"),
                    # dict(type="CenterShift", apply_z=False),
                    dict(type="Offset_nusc", condition="nuScenes"),
                    dict(type="Add", keys_dict={"condition": "nuScenes"}), #add condition 
                    dict(type="ToTensor"),
                    dict(type="Collect",
                         keys=("coord", "grid_coord", "discrete_coord", "segment", "condition",  "pairing_points", "pairing_images", "inverse_indexes","sam_features","img_logits", "gt_off", "inst_data","valid", "org_coord"),
                         feat_keys=("coord", "strength"))
                ],
                test_mode=False,
                ignore_index=-1,
                loop=1
            ),
            # SemanticKITTI
            dict(
                type="SemanticKITTIDataset",
                split="train",
                data_root="/data/sets/semantickitti/sequences/",
                transform=[
                    # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
                    # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
                    # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="x", p=0.5),
                    # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="y", p=0.5),
                    # dict(type="PointClip", point_cloud_range=(-50, -50, -4, 50, 50, 2)),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                    dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train",
                         keys=("coord", "strength", "segment"), return_grid_coord=True),
                    # dict(type="SphereCrop", point_max=1000000, mode="random"),
                    # dict(type="CenterShift", apply_z=False),
                    dict(type="Offset_kitti", condition="SemanticKITTI"),
                    dict(type="Add", keys_dict={"condition": "SemanticKITTI"}),
                    dict(type="ToTensor"),
                    dict(type="Collect",
                         keys=("coord", "grid_coord", "discrete_coord", "segment", "condition","pairing_points", "pairing_images", "inverse_indexes","sam_features","img_logits", "gt_off", "inst_data","valid", "org_coord"),
                         feat_keys=("coord", "strength"))
                ],
                test_mode=False,
                ignore_index=-1,
                loop=1
            ),
            # Waymo
            dict(
                type="WaymoDataset",
                split="training",
                data_root="/data/sets/waymo",
                transform=[
                    # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
                    # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
                    # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="x", p=0.5),
                    # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="y", p=0.5),
                    # dict(type="PointClip", point_cloud_range=(-75.2, -75.2, -4, 75.2, 75.2, 2)),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                    dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train",
                         keys=("coord", "strength", "segment"), return_discrete_coord=True),
                    # dict(type="SphereCrop", point_max=1000000, mode="random"),
                    # dict(type="CenterShift", apply_z=False),
                    dict(type="Add", keys_dict={"condition": "Waymo"}),
                    dict(type="ToTensor"),
                    dict(type="Collect",
                         keys=("coord", "grid_coord", "discrete_coord", "segment", "condition"),
                         feat_keys=("coord", "strength"))
                ],
                test_mode=False,
                ignore_index=-1,
                loop=1
            ),
        ]
    ),
    val=dict(
        type="WaymoDataset",
        split="val",
        data_root="/data/sets/waymo",
        transform=[
            # dict(type="PointClip", point_cloud_range=(-51.2, -51.2, -4, 51.2, 51.2, 2.4)),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment"),
                return_grid_coord=True,
            ),
            # dict(type="Offset_kitti", condition="SemanticKITTI"),
            dict(type="Add", keys_dict={"condition": "Waymo"}),
            # dict(type="SphereCrop", point_max=1000000, mode='center'),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "condition"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    test=dict(
        type="WaymoDataset",
        split="val",
        data_root="/data/sets/waymo",
        transform=[
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.025,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment"),
                return_inverse=True,
            ),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                keys=("coord", "strength"),
            ),
            crop=None,
            post_transform=[
                dict(type="Add", keys_dict={"condition": "Waymo"}),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index", "condition"),
                    feat_keys=("coord", "strength"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.1, 1.1])],
                [
                    dict(type="RandomScale", scale=[0.9, 0.9]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                    dict(type="RandomFlip", p=1),
                ],
                [dict(type="RandomScale", scale=[1, 1]), dict(type="RandomFlip", p=1)],
                [
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1.1, 1.1]),
                    dict(type="RandomFlip", p=1),
                ],
            ],
            panoptic_mode=False,
        ),
        ignore_index=ignore_index,
    ),
)