_base_ = [
    '_base_svtr-tiny.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_base.py',
]

# from svtr-small_20e_st_jm.py
model = dict(
    encoder=dict(
        embed_dims=[96, 192, 256],
        depth=[3, 6, 6],
        num_heads=[3, 6, 8],
        mixer_types=['Local'] * 8 + ['Global'] * 7))


# svtr-tiny_20e_st_mj.py settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=60, val_interval=1)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=5 / (10**4) * 2048 / 2048,
        betas=(0.9, 0.99),
        eps=8e-8,
        weight_decay=0.05))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.5,
        end_factor=1.,
        end=2,
        verbose=False,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=19,
        begin=2,
        end=20,
        verbose=False,
        convert_to_iter_based=True),
]

# my additions

skip_cat_dictionary = dict(
        type="Dictionary",
        dict_file='{{ fileDirname }}/../../../dicts/SKIP_categories.txt',
        with_padding=True,
        with_unknown=True,
)

relaxed_brsk_dictionary = dict(
        type="Dictionary",
        dict_file='{{ fileDirname }}/../../../dicts/relaxed_brsk.txt',
        with_padding=True,
        with_unknown=True,
)

specialized_dictionary = dict(
        type="Dictionary",
        dict_file='{{ fileDirname }}/../../../dicts/SKIP_category/', # needs to append the correct filename at runtime
        with_padding=True,
        with_unknown=True,
)


kuzushiji_textrecog_data_root = ""

img_path = "/kaggle/input/kuzushiji-characters/"

kuzushiji_textrecog_train = dict(
        type="OCRDataset",
        data_root=kuzushiji_textrecog_data_root,
        ann_file="configs/textrecog/textrecog_train_kuzushiji.json",
        data_prefix=dict(img_path=img_path),
        pipeline=None)

train_list = [kuzushiji_textrecog_train]

kuzushiji_textrecog_test = dict(
        type="OCRDataset",
        data_root=kuzushiji_textrecog_data_root,
        ann_file="configs/textrecog/textrecog_validation_kuzushiji.json",
        data_prefix=dict(img_path=img_path),
        test_mode=True,
        pipeline=None)

test_list = [kuzushiji_textrecog_test]

val_evaluator = dict(
        dataset_prefixes=['字'],
        metrics=[dict(type="CharMetric")]
    )
test_evaluator = val_evaluator

load_from="/kaggle/working/svtr-small_20e_st_mj-35d800d6.pth"

train_pipeline = [
    dict(type='LoadImageFromFile', ignore_empty=True, min_size=1),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='RandomApply',
        prob=0.4,
        transforms=[
            dict(type='TextRecogGeneralAug', ),
        ],
    ),
    dict(
        type='RandomApply',
        prob=0.4,
        transforms=[
            dict(type='CropHeight', ),
        ],
    ),
    dict(
        type='ConditionApply',
        condition='min(results["img_shape"])>10',
        true_transforms=dict(
            type='RandomApply',
            prob=0.4,
            transforms=[
                dict(
                    type='TorchVisionWrapper',
                    op='GaussianBlur',
                    kernel_size=5,
                    sigma=1,
                ),
            ],
        )),
    dict(
        type='RandomApply',
        prob=0.4,
        transforms=[
            dict(
                type='TorchVisionWrapper',
                op='ColorJitter',
                brightness=0.5,
                saturation=0.5,
                contrast=0.5,
                hue=0.1),
        ]),
    dict(
        type='RandomApply',
        prob=0.4,
        transforms=[
            dict(type='ImageContentJitter', ),
        ],
    ),
    dict(
        type='RandomApply',
        prob=0.4,
        transforms=[
            dict(
                type='ImgAugWrapper',
                args=[dict(cls='AdditiveGaussianNoise', scale=0.1**0.5)]),
        ],
    ),
    dict(
        type='RandomApply',
        prob=0.4,
        transforms=[
            dict(type='ReversePixels', ),
        ],
    ),
    dict(type='Resize', scale=(256, 64)),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

# from _base_svtr-tiny.py
train_dataloader = dict(
    batch_size=384,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=128,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=test_list,
        pipeline=_base_.test_pipeline))

test_dataloader = val_dataloader
