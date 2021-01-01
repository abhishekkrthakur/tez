import cv2
import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class RCNNDataset:
    def __init__(self, image_paths, bounding_boxes, classes=None, augmentations=None, torchvision_format=True):
        self.image_paths = image_paths
        self.bounding_boxes = bounding_boxes
        self.augmentations = augmentations
        self.torchvision_format = torchvision_format
        self.classes = classes

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        # image = Image.open(self.image_paths[item])
        image = cv2.imread(self.image_paths[item])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = self.bounding_boxes[item]
        image = np.array(image)
        if self.augmentations is not None:
            augmented = self.augmentations(image=image, bboxes=bboxes)
            image = augmented["image"]
            bboxes = augmented["bboxes"]

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        bboxes = np.array(bboxes)

        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])

        is_crowd = torch.zeros((bboxes.shape[0],), dtype=torch.int64)
        if self.classes is None:
            labels = torch.ones((bboxes.shape[0],), dtype=torch.int64)
        else:
            labels = torch.tensor(self.classes[item], dtype=torch.int64)

        target = {
            "boxes": torch.as_tensor(bboxes.tolist(), dtype=torch.float32),
            "area": torch.as_tensor(area.tolist(), dtype=torch.float32),
            "iscrowd": is_crowd,
            "labels": labels,
        }

        if self.torchvision_format:
            return torch.tensor(image, dtype=torch.float), target

        target["image"] = torch.tensor(image, dtype=torch.float)
        return target
