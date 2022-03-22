# For dataset, go to: https://www.kaggle.com/msheriey/104-flowers-garden-of-eden
# Update INPUT_PATH and MODEL_PATH
import glob
import os

import albumentations
import numpy as np
import pandas as pd
import timm
import torch.nn as nn

from tez import Tez, TezConfig
from tez.datasets import ImageDataset


INPUT_PATH = "../../input/"
MODEL_PATH = "../../models/"
MODEL_NAME = "flower_classification"
IMAGE_SIZE = 192

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

    def forward(self, image, targets=None):
        outputs = self.model(image)
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
    model = Tez(model)
    model_path = os.path.join(MODEL_PATH, MODEL_NAME + ".bin")
    config = TezConfig(
        test_batch_size=64,
        device="cuda",
    )
    model.load(model_path, weights_only=True, config=config)

    preds_iter = model.predict(test_dataset)
    final_preds = []
    for preds in preds_iter:
        final_preds.append(preds)
    final_preds = np.vstack(final_preds)
    final_preds = np.argmax(final_preds, axis=1)

    test_image_names = [x.split("/")[-1][:-5] for x in test_image_paths]
    df = pd.DataFrame({"id": test_image_names, "label": final_preds})
    df.to_csv("result.csv", index=False)
