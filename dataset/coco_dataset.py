import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


class COCODataset(Dataset):
    """
    A custom dataset class for the multi-task Scene+Projection model.
    This class is specifically designed to load data from a .jsonl annotation file
    where each line is a JSON object containing lists of captions.
    """

    def __init__(self, data_root):
        """
        Args:
            data_root (str): The root directory of the dataset, which should contain
                             'images', 'masks', and 'annotations' subdirectories.
            ann_file_name (str): The name of the .jsonl annotation file inside the 'annotations' folder.
        """
        ann_path = os.path.join(data_root, 'train_coco_scenes_1_to_60.jsonl')
        self.data_root = data_root

        # Read .jsonl file line by line ---
        print(f"Loading annotations from .jsonl file: {ann_path}")
        self.annotations = []
        with open(ann_path, "r", encoding="utf-8") as f:
            for line in f:
                self.annotations.append(json.loads(line.strip()))

        # Define image transformations
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

        # Define mask transformations
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((64, 64), interpolation=InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index):
        ann = self.annotations[index]

        # Load and transform the image
        image_path = os.path.join(self.data_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        # Load and transform the ground truth mask
        mask_path = os.path.join(self.data_root, ann["mask"])
        gt_mask = Image.open(mask_path).convert("L")
        gt_mask = self.mask_transform(gt_mask)

        # Randomly select one caption from the list for training
        scene_caption = random.choice(ann["scene_captions"])
        projection_caption = random.choice(ann["projection_captions"])

        # Return the complete dictionary required by the model
        return {
            "image": image,
            "scene_caption": scene_caption,
            "projection_caption": projection_caption,
            "gt_mask": gt_mask,
        }
