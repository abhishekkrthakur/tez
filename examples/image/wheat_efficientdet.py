# fetch data from: https://www.kaggle.com/competitions/global-wheat-detection

import argparse
import ast
import os

import albumentations
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torchvision
from effdet import DetBenchTrain, EfficientDet, get_efficientdet_config
from effdet.config.model_config import efficientdet_model_param_dict
from effdet.efficientdet import HeadNet
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from tez import Tez, TezConfig


ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    """parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=0.005, required=False)
    parser.add_argument("--batch_size", type=int, default=8, required=False)
    parser.add_argument("--epochs", type=int, default=10, required=False)
    parser.add_argument("--output", type=str, default=".")
    parser.add_argument("--accumulation_steps", type=int, default=1, required=False)
    return parser.parse_args()


def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)


def collate_fn(batch):
    return tuple(zip(*batch))


def create_model(num_classes=1, image_size=512, architecture="tf_efficientnetv2_l"):
    efficientdet_model_param_dict["tf_efficientnetv2_l"] = dict(
        name="tf_efficientnetv2_l",
        backbone_name="tf_efficientnetv2_l",
        backbone_args=dict(drop_path_rate=0.2),
        num_classes=num_classes,
        url="",
    )

    config = get_efficientdet_config(architecture)
    config.update({"num_classes": num_classes})
    config.update({"image_size": (image_size, image_size)})

    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
    )
    return DetBenchTrain(net, config)


class WheatDataset:
    def __init__(self, image_paths, bounding_boxes, augmentations=None):
        self.image_paths = image_paths
        self.bounding_boxes = bounding_boxes
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item])
        bboxes = self.bounding_boxes[item]

        image = np.array(image)
        if self.augmentations is not None:
            augmented = self.augmentations(image=image, bboxes=bboxes)
            image = augmented["image"]
            bboxes = augmented["bboxes"]

        bboxes = [b[:4] for b in bboxes]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        bboxes = np.array(bboxes)

        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]

        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        labels = torch.ones((bboxes.shape[0],), dtype=torch.int64)
        is_crowd = torch.zeros((bboxes.shape[0],), dtype=torch.int64)

        _, new_h, new_w = image.shape

        target = {
            "bboxes": torch.as_tensor(bboxes.tolist(), dtype=torch.float32),
            "area": torch.as_tensor(area.tolist(), dtype=torch.float32),
            "iscrowd": is_crowd,
            "labels": labels,
            "img_size": torch.tensor([new_h, new_w], dtype=torch.int64),
            "img_scale": torch.tensor([1.0]),
        }

        image = torch.tensor(image, dtype=torch.float)

        return image, target


class WheatModel(nn.Module):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.base_model = create_model(num_classes=1, image_size=1024, architecture="tf_efficientdet_d1")

    def optimizer_scheduler(self):
        opt = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=0.0005,
        )
        # define scheduler here if needed
        return opt, None

    def forward(self, images, targets):
        outputs = self.base_model(images, targets)
        if targets is not None:
            loss = outputs["loss"]
            return outputs, loss, {}
        return outputs, 0, {}


class TezEfficientDet(Tez):
    def __init__(self, model):
        super().__init__(model)

    def model_fn(self, data):
        images, targets = data
        images = list(image.to(self.config.device) for image in images)
        images = torch.stack(images)
        images = images.float()
        targets = [{k: v.to(self.config.device) for k, v in t.items()} for t in targets]

        boxes = [target["bboxes"].float() for target in targets]
        labels = [target["labels"].float() for target in targets]
        img_size = torch.stack([target["img_size"] for target in targets]).float()
        img_scale = torch.stack([target["img_scale"] for target in targets]).float()

        annotations = {
            "bbox": boxes,
            "cls": labels,
            "img_size": img_size,
            "img_scale": img_scale,
        }

        if self.config.fp16:
            with torch.cuda.amp.autocast():
                output, loss, metrics = self.model(images, annotations)
        else:
            output, loss, metrics = self.model(images, annotations)
        return output, loss, metrics


if __name__ == "__main__":
    args = parse_args()
    df = pd.read_csv(os.path.join(args.input, "train.csv"))  # .head(100)
    df.bbox = df.bbox.fillna("[0, 0, 10, 10]")
    df.bbox = df.bbox.apply(ast.literal_eval)
    df = df.groupby("image_id")["bbox"].apply(list).reset_index(name="bboxes")

    images = df.image_id.values.tolist()
    images = [os.path.join(args.input, "train", i + ".jpg") for i in images]
    targets = df.bboxes.values
    for target in targets:
        for sub_target in target:
            sub_target.append("wheat")

    train_images, val_images, train_targets, val_targets = train_test_split(
        images, targets, test_size=0.1, random_state=42
    )

    mean = (0.0, 0.0, 0.0)
    std = (1, 1, 1)
    aug = albumentations.Compose(
        [
            albumentations.Normalize(
                mean,
                std,
                max_pixel_value=255.0,
                always_apply=True,
            )
        ],
        bbox_params=albumentations.BboxParams(format="coco"),
    )

    train_dataset = WheatDataset(
        image_paths=train_images,
        bounding_boxes=train_targets,
        augmentations=aug,
    )
    valid_dataset = WheatDataset(
        image_paths=val_images,
        bounding_boxes=val_targets,
        augmentations=aug,
    )

    model = WheatModel(learning_rate=args.learning_rate)
    model = TezEfficientDet(model)

    config = TezConfig(
        training_batch_size=args.batch_size,
        validation_batch_size=2 * args.batch_size,
        test_batch_size=2 * args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        epochs=args.epochs,
        fp16=True,
    )
    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        train_collate_fn=collate_fn,
        valid_collate_fn=collate_fn,
        config=config,
    )
