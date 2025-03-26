import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torchvision import transforms
from torchinfo import summary

from engine import train_model
from utils import set_seeds, download_data, plot_loss_curves, save_model
from data_setup import create_dataloaders
from model import VisionTransformer
from fetch_data import create_dir_structue


# print(torch.__version__)
# print(torchvision.__version__)

# Hyperparameters
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
PATCH_SIZE = 16
NUM_CLASSES = 3
EMB_SIZE = 768
DEPTH = 12
NUM_HEADS = 12
MLP_DIM = 3072
DROPOUT = 0.1
LR = 3e-3
EPOCHS = 10
WEIGHT_DECAY = 0.3
BETAS = (0.9, 0.999)


device = torch.device('mps')

# Set the seed
set_seeds(seed=SEED)

# Download the data
image_path = create_dir_structue()
print(f"Data path: {image_path}")

train_dir = image_path / "train"
test_dir = image_path / "test"

# Define the transforms
simple_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# Load the data
train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=train_dir,
                                                                    test_dir=test_dir,
                                                                    batch_size=BATCH_SIZE,
                                                                    transfrom=simple_transforms)


# Load the model
model = VisionTransformer(image_size=IMG_SIZE,
                          patch_size=PATCH_SIZE,
                          num_classes=NUM_CLASSES,
                          emb_size=EMB_SIZE,
                          depth=DEPTH,
                          num_heads=NUM_HEADS,
                          mlp_dim=MLP_DIM,
                          dropout=DROPOUT).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
loss_fn = nn.CrossEntropyLoss()

# Summary of the model
summary(model, 
        input_size=(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE),
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds", "trainable"],
        row_settings=("var_names"))

# Train the model
results = train_model(model=model,
                      optimizer=optimizer,
                      loss_fn=loss_fn,
                      train_dataloader=train_dataloader,
                      test_dataloader=test_dataloader,
                      device=device,
                      epochs=EPOCHS)

save_model(model=model, path="models", model_name="vision_transformer.pth")