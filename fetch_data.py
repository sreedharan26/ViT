import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms
import pathlib
import random
import shutil

data_dir = pathlib.Path("./data")

train_data = datasets.Food101(root=data_dir,
                              split='train',
                              download=True)

test_data = datasets.Food101(root=data_dir,
                             split='test',
                             download=True)

class_names = train_data.classes

data_path = data_dir / "food-101" / "images"
target_classes = ["pizza", "steak", "sushi"]

AMOUNT_TO_GET = 0.4

def create_subset(image_path=data_path,
                  data_splits=["train", "test"],
                  target_classes=target_classes,
                  amount_to_get=AMOUNT_TO_GET,
                  seed=42):
    random.seed(seed)

    label_splits = {}

    for split in data_splits:
        label_path = data_dir / "food-101" / "meta" / f"{split}.txt"
        with open(label_path, "r") as f:
            labels = [line.strip('\n') for line in f.readlines() if line.split("/")[0] in target_classes]

        number_to_sample = round(AMOUNT_TO_GET * len(labels))
        print(f"[INFO] Getting random subset of {number_to_sample} images for {split}...")
        sampled_images = random.sample(labels, k=number_to_sample)
        
        image_paths = [pathlib.Path(str(image_path / sample_image) + ".jpg") for sample_image in sampled_images]
        label_splits[split] = image_paths
    return label_splits


def create_dir_structue():
    label_splits = create_subset()

    target_dir = f"./data/{'_'.join(target_classes)}_{str(int(AMOUNT_TO_GET*100))}_percent"
    target_dir = pathlib.Path(target_dir)
    if target_dir.exists():
        print(f"[INFO] Directory {target_dir} already exists.")
        return target_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    for image_split in label_splits.keys():
        for image_path in label_splits[image_split]:
            dest_dir = target_dir / image_split / image_path.parent.name / image_path.name
            if not dest_dir.parent.is_dir():
                dest_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(image_path, dest_dir)
    print(f"[INFO] Done! Images are saved to {target_dir}")
    return target_dir