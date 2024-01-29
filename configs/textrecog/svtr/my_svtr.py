_base_ = ['svtr-small_20e_st_mj.py']

dictionary = dict(
        type="Dictionary",
        dict_file='{{ fileDirName }}/../../../dicts/SKIP_categories.txt',
        with_padding=True,
        with_unknown=True,
)

kuzushiji_textrecog_data_root = "/kaggle/input/bsrk-recognition/resized_train_images/"

kuzushiji_textrecog_train = dict(
        type="OCRDataset",
        data_root=kuzushiji_textrecog_data_root,
        ann_file="../ocr_dataset_resized_train.json",
        pipeline=None)

train_list = [kuzushiji_textrecog_train]

kuzushiji_textrecog_test = dict(
        type="OCRDataset",
        data_root=kuzushiji_textrecog_data_root,
        ann_file="../ocr_dataset_resized_validation.json",
        test_mode=True,
        pipeline=None)

test_list = [kuzushiji_textrecog_test]

val_evaluator = dict(dataset_prefixes=['字'])
test_evaluator = val_evaluator
