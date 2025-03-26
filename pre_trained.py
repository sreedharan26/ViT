import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchinfo import summary

from engine import train_model
from utils import set_seeds, download_data, plot_loss_curves, save_model
from data_setup import create_dataloaders
from fetch_data import create_dir_structue

device = torch.device('mps')

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

# Load the pre-trained model
pre_trained_model_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
pre_trained_model = torchvision.models.vit_b_16(weights=pre_trained_model_weights).to(device)

#  Freeze the model
for param in pre_trained_model.parameters():
    param.requires_grad = False

# Set the seed
set_seeds(seed=SEED)

# Replace the final layer
pre_trained_model.heads = nn.Sequential(
    nn.LayerNorm(EMB_SIZE),
    nn.Linear(in_features=EMB_SIZE, out_features=NUM_CLASSES),
).to(device)

# Summary of the model
summary(pre_trained_model, input_size=(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE), col_names=["input_size", "output_size", "num_params", "trainable"], row_settings=("var_names",))

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


# Loss function
loss_fn = nn.CrossEntropyLoss()
# Optimizer
optimizer = torch.optim.Adam(pre_trained_model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)

# Train the model
results = train_model(model=pre_trained_model.to(device),
                      optimizer=optimizer,
                      loss_fn=loss_fn,
                      train_dataloader=train_dataloader,
                      test_dataloader=test_dataloader,
                      device=device,
                      epochs=EPOCHS)

# Save the model
save_model(model=pre_trained_model, path='models', model_name="pre_trained_model.pth")

# Plot the loss curves
plot_loss_curves(results)