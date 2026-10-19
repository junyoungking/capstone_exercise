"""YOLOX-Tiny, one class "plate" (구현 설계 tab, 모델 학습). Used by YOLOX's tools/train.py and export_onnx.py.

Data folder from the PLATE_DATA environment variable (default datasets/plate), made by
train.plate.make_dataset.
"""
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        # YOLOX-Tiny
        self.depth = 0.33
        self.width = 0.375
        self.input_size = (416, 416)
        self.test_size = (416, 416)
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        self.enable_mixup = False
        # data
        self.num_classes = 1
        self.data_dir = os.environ.get("PLATE_DATA", "datasets/plate")
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        # augmentation: size from mosaic/scale, brightness from HSV, small rotations (camera tilt)
        self.mosaic_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 5.0
        self.translate = 0.1
        self.shear = 2.0
        # schedule (fine-tuning from yolox_tiny.pth)
        self.max_epoch = int(os.environ.get("PLATE_EPOCHS", 80))
        self.no_aug_epochs = 10
        self.warmup_epochs = 3
        self.basic_lr_per_img = 0.01 / 64.0
        self.eval_interval = 5
        self.data_num_workers = int(os.environ.get("PLATE_WORKERS", 4))
        self.exp_name = "plate_tiny"
