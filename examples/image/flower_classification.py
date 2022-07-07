# For dataset, go to: https://www.kaggle.com/msheriey/104-flowers-garden-of-eden
# Update INPUT_PATH and MODEL_PATH
import glob
import os

import albumentations
import timm
import torch
import torch.nn as nn
from sklearn import metrics

from tez import Tez, TezConfig
from tez.callbacks import EarlyStopping
from tez.datasets import ImageDataset


INPUT_PATH = "data/"
MODEL_PATH = "data/models/"
MODEL_NAME = os.path.basename(__file__)[:-3]
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
IMAGE_SIZE = 192
EPOCHS = 200

CLASSES = {
    "pink primrose": 0,
    "hard-leaved pocket orchid": 1,
    "canterbury bells": 2,
    "sweet pea": 3,
    "wild geranium": 4,
    "tiger lily": 5,
    "moon orchid": 6,
    "bird of paradise": 7,
    "monkshood": 8,
    "globe thistle": 9,
    "snapdragon": 10,
    "colt's foot": 11,
    "king protea": 12,
    "spear thistle": 13,
    "yellow iris": 14,
    "globe-flower": 15,
    "purple coneflower": 16,
    "peruvian lily": 17,
    "balloon flower": 18,
    "giant white arum lily": 19,
    "fire lily": 20,
    "pincushion flower": 21,
    "fritillary": 22,
    "red ginger": 23,
    "grape hyacinth": 24,
    "corn poppy": 25,
    "prince of wales feathers": 26,
    "stemless gentian": 27,
    "artichoke": 28,
    "sweet william": 29,
    "carnation": 30,
    "garden phlox": 31,
    "love in the mist": 32,
    "cosmos": 33,
    "alpine sea holly": 34,
    "ruby-lipped cattleya": 35,
    "cape flower": 36,
    "great masterwort": 37,
    "siam tulip": 38,
    "lenten rose": 39,
    "barberton daisy": 40,
    "daffodil": 41,
    "sword lily": 42,
    "poinsettia": 43,
    "bolero deep blue": 44,
    "wallflower": 45,
    "marigold": 46,
    "buttercup": 47,
    "daisy": 48,
    "common dandelion": 49,
    "petunia": 50,
    "wild pansy": 51,
    "primula": 52,
    "sunflower": 53,
    "lilac hibiscus": 54,
    "bishop of llandaff": 55,
    "gaura": 56,
    "geranium": 57,
    "orange dahlia": 58,
    "pink-yellow dahlia": 59,
    "cautleya spicata": 60,
    "japanese anemone": 61,
    "black-eyed susan": 62,
    "silverbush": 63,
    "californian poppy": 64,
    "osteospermum": 65,
    "spring crocus": 66,
    "iris": 67,
    "windflower": 68,
    "tree poppy": 69,
    "gazania": 70,
    "azalea": 71,
    "water lily": 72,
    "rose": 73,
    "thorn apple": 74,
    "morning glory": 75,
    "passion flower": 76,
    "lotus": 77,
    "toad lily": 78,
    "anthurium": 79,
    "frangipani": 80,
    "clematis": 81,
    "hibiscus": 82,
    "columbine": 83,
    "desert-rose": 84,
    "tree mallow": 85,
    "magnolia": 86,
    "cyclamen ": 87,
    "watercress": 88,
    "canna lily": 89,
    "hippeastrum ": 90,
    "bee balm": 91,
    "pink quill": 92,
    "foxglove": 93,
    "bougainvillea": 94,
    "camellia": 95,
    "mallow": 96,
    "mexican petunia": 97,
    "bromelia": 98,
    "blanket flower": 99,
    "trumpet creeper": 100,
    "blackberry lily": 101,
    "common tulip": 102,
    "wild rose": 103,
}


class FlowerModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model("resnet18", pretrained=True)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, num_classes)

    def monitor_metrics(self, outputs, targets):
        device = targets.device.type
        outputs = torch.argmax(outputs, dim=1).cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        f1 = metrics.f1_score(targets, outputs, average="macro")
        return {"f1": torch.tensor(f1, device=device)}

    def optimizer_scheduler(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=0.5,
            patience=2,
            verbose=True,
            mode="max",
            threshold=1e-4,
        )
        return opt, sch

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

    test_aug = albumentations.Compose(
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

    test_image_paths = glob.glob(
        os.path.join(INPUT_PATH, f"jpeg-{IMAGE_SIZE}x{IMAGE_SIZE}", "test", "*.jpeg"),
    )

    train_targets = [x.split("/")[-2] for x in train_image_paths]
    valid_targets = [x.split("/")[-2] for x in valid_image_paths]

    train_targets = [CLASSES[c] for c in train_targets]
    valid_targets = [CLASSES[c] for c in valid_targets]

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

    test_dataset = ImageDataset(
        image_paths=test_image_paths,
        targets=[0] * len(test_image_paths),
        augmentations=test_aug,
    )

    model = FlowerModel(num_classes=len(CLASSES))
    es = EarlyStopping(
        monitor="valid_f1",
        model_path=os.path.join(MODEL_PATH, MODEL_NAME + ".bin"),
        patience=7,
        mode="max",
        save_weights_only=True,
    )
    model = Tez(model)
    config = TezConfig(
        training_batch_size=TRAIN_BATCH_SIZE,
        validation_batch_size=VALID_BATCH_SIZE,
        epochs=EPOCHS,
        step_scheduler_after="epoch",
        step_scheduler_metric="valid_f1",
        val_strategy="batch",
        val_steps=200,
    )
    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        config=config,
        callbacks=[es],
    )
