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
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)

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

kuzushiji_textrecog_data_root = ""

img_path = "/kaggle/input/cropped-kuzushiji-characters/character_images/"

kuzushiji_textrecog_train = dict(
        type="OCRDataset",
        data_root=kuzushiji_textrecog_data_root,
        ann_file="configs/textrecog/ocr_dataset_resized_train.json",
        data_prefix=dict(img_path=img_path),
        pipeline=None)

train_list = [kuzushiji_textrecog_train]

kuzushiji_textrecog_test = dict(
        type="OCRDataset",
        data_root=kuzushiji_textrecog_data_root,
        ann_file="configs/textrecog/ocr_dataset_resized_validation.json",
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


# from _base_svtr-tiny.py
train_dataloader = dict(
    batch_size=256,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline))

val_dataloader = dict(
    batch_size=64,
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
