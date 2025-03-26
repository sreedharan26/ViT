import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import os
import zipfile
import requests
from pathlib import Path

def set_seeds(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def download_data(url: str, path: str, remove_source: bool = True) -> Path:
    
    data_path = Path("data/")
    image_path = data_path / path

    if image_path.is_dir():
        print(f"[INFO] Data already downloaded at {image_path}")
    else:
        print(f"[INFO] Did not find data at {image_path}, creating directory")
        image_path.mkdir(parents=True, exist_ok=True)

        target_file = Path(url).name
        with open(data_path / target_file, "wb") as file:
            print(f"Downloading {target_file} from {url}")
            response = requests.get(url)
            file.write(response.content)
        
        print(f"[INFO] Extracting {target_file}")
        with zipfile.ZipFile(data_path / target_file, 'r') as zip_ref:
            zip_ref.extractall(image_path)
        
        if remove_source:
            os.remove(data_path / target_file)
        
    return image_path


def plot_loss_curves(results):

    train_loss = results["train_loss"]
    test_loss = results["test_loss"]

    train_accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]  

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="train_loss")
    plt.plot(test_loss, label="test_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, label="train_accuracy")
    plt.plot(test_accuracy, label="test_accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()  


def save_model(model: nn.Module, path: str, model_name: str):
    target_path = Path(path)
    target_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith((".pt", ".pth")), "Model name should end with .pt or .pth"
    torch.save(obj=model.state_dict(), f=target_path / model_name)
    print(f"[INFO] Model saved at {target_path / model_name}")