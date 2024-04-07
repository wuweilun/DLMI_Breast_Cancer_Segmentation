import argparse
import os
import sys
import warnings
import glob
import pandas as pd
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from sklearn.model_selection import train_test_split
from utils import CustomDataset, bce_dice_loss, Trainer, plot_metrics, plot_subplots
import segmentation_models_pytorch as smp

def get_model(model_name):
    if model_name == "unet":
        return smp.Unet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=1, classes=1)
    elif model_name == "unet++":
        return smp.UnetPlusPlus(encoder_name="resnet50", encoder_weights="imagenet", in_channels=1, classes=1)
    elif model_name == "manet":
        return smp.MAnet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=1, classes=1)
    elif model_name == "linknet":
        return smp.Linknet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=1, classes=1)
    elif model_name == "fpn":
        return smp.FPN(encoder_name="resnet50", encoder_weights="imagenet", in_channels=1, classes=1)
    elif model_name == "pspnet":
        return smp.PSPNet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=1, classes=1)
    elif model_name == "pan":
        return smp.PAN(encoder_name="resnet50", encoder_weights="imagenet", in_channels=1, classes=1)
    elif model_name == "deeplabv3":
        return smp.DeepLabV3(encoder_name="resnet50", encoder_weights="imagenet", in_channels=1, classes=1)
    elif model_name == "deeplabv3+":
        return smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights="imagenet", in_channels=1, classes=1)
    else:
        raise ValueError("Model name not supported!")
    
model_name = sys.argv[1]
image_path = sys.argv[2]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

masks = glob.glob("Dataset_BUSI_with_GT/*/*_mask.png")
images = [mask_images.replace("_mask", "") for mask_images in masks]
dataset = pd.DataFrame(list(zip(images, masks)), columns=['image_path', 'mask_path'])
train, test= train_test_split(dataset, test_size=0.2)

image_size = 224
    
train_transforms = transforms.Compose([
    transforms.Resize([image_size, image_size]),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize([image_size, image_size]),
    transforms.ToTensor(),
])

train_dataset = CustomDataset(train, train_transforms)
test_dataset = CustomDataset(test, val_transforms)

batch_size = 8
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = get_model(model_name).to(device)
epochs = 100
learning_rate = 0.0001
weight_decay = 1e-6

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

trainer = Trainer(model=model, model_name=model_name, num_epochs=epochs, optimizer=optimizer, criterion=bce_dice_loss, device=device, project_name="DLMI_HW")

trainer.train(train_dataloader, test_dataloader)

# plot_metrics(metrics=trainer.get_metrics(), image_path=image_path, model_name=model_name)

# for i in [2, 3, 11, 20, 55, 67, 87, 98, 120, 130, 200]:
#     image = train_dataset[i][0]
#     mask = train_dataset[i][1]
#     image = image.to(device)
#     pred = model(image.unsqueeze(0))
#     pred = pred.squeeze()

#     plot_subplots(image, mask, pred, 0.5, image_path, model_name, i)