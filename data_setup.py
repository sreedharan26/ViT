import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

no_of_workers = os.cpu_count()

def create_dataloaders(train_dir: str, 
                       test_dir: str, 
                       batch_size: int = 32, 
                    #    num_workers: int = no_of_workers,
                       transfrom: transforms.Compose = None):
    """
    Create dataloaders for training and testing datasets
    """
    train_data = datasets.ImageFolder(root=train_dir, transform=transfrom)
    test_data = datasets.ImageFolder(root=test_dir, transform=transfrom)

    class_names = train_data.classes

    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                #   num_workers=num_workers)
                                  )   
    test_dataloader = DataLoader(test_data,
                                 batch_size=batch_size, 
                                 shuffle=False, 
                                #  num_workers=num_workers)
                                 )
    
    return train_dataloader, test_dataloader, class_names