# For dataset, go to: https://www.kaggle.com/msheriey/104-flowers-garden-of-eden
# Update INPUT_PATH and MODEL_PATH
import glob
import os

import albumentations
import timm
import torch
import torch.nn as nn
from sklearn import metrics, preprocessing
from torch.nn import functional as F

from tez import Tez, TezConfig
from tez.callbacks import EarlyStopping
from tez.datasets import ImageDataset


INPUT_PATH = "../../input/"
MODEL_PATH = "../../models/"
MODEL_NAME = os.path.basename(__file__)[:-3]
TRAIN_BATCH_SIZE = 512
VALID_BATCH_SIZE = 32
IMAGE_SIZE = 192
EPOCHS = 20


class FlowerModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model("resnet18", pretrained=True)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, num_classes)

    def monitor_metrics(self, outputs, targets):
        device = targets.get_device()
        outputs = torch.argmax(outputs, dim=1).cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(targets, outputs)
        return {"accuracy": torch.tensor(accuracy, device=device)}

    def fetch_optimizer(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        return opt

    def forward(self, image, targets=None):
        outputs = self.model(image)
        if targets is not None:
            loss = nn.CrossEntropyLoss()(outputs, targets)
            metrics = self.monitor_metrics(outputs, targets)
            return outputs, loss, metrics
        return outputs, 0, {}


if __name__ == "__main__":
    train_aug = albumentations.Compose(
        [
            albumentations.Transpose(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.ShiftScaleRotate(p=0.5),
            albumentations.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            albumentations.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ],
        p=1.0,
    )

    valid_aug = albumentations.Compose(
        [
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ],
        p=1.0,
    )

    train_image_paths = glob.glob(
        os.path.join(INPUT_PATH, f"jpeg-{IMAGE_SIZE}x{IMAGE_SIZE}", "train", "**", "*.jpeg"),
        recursive=True,
    )

    valid_image_paths = glob.glob(
        os.path.join(INPUT_PATH, f"jpeg-{IMAGE_SIZE}x{IMAGE_SIZE}", "val", "**", "*.jpeg"),
        recursive=True,
    )

    train_targets = [x.split("/")[-2] for x in train_image_paths]
    valid_targets = [x.split("/")[-2] for x in valid_image_paths]

    lbl_enc = preprocessing.LabelEncoder()
    train_targets = lbl_enc.fit_transform(train_targets)
    valid_targets = lbl_enc.transform(valid_targets)

    train_dataset = ImageDataset(
        image_paths=train_image_paths,
        targets=train_targets,
        augmentations=train_aug,
    )

    valid_dataset = ImageDataset(
        image_paths=valid_image_paths,
        targets=valid_targets,
        augmentations=valid_aug,
    )

    model = FlowerModel(num_classes=len(lbl_enc.classes_))
    es = EarlyStopping(
        monitor="valid_accuracy",
        model_path=os.path.join(MODEL_PATH, MODEL_NAME + ".bin"),
        patience=3,
        mode="max",
    )
    model = Tez(model)
    config = TezConfig(
        training_batch_size=TRAIN_BATCH_SIZE,
        validation_batch_size=VALID_BATCH_SIZE,
        epochs=EPOCHS,
    )
    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        config=config,
        callbacks=[es],
    )
