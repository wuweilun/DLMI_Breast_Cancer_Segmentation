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
from utils import CustomDataset, bce_dice_loss, Trainer, plot_metrics, plot_subplots, plot_predictions
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
    
image_path = sys.argv[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

masks = glob.glob("Dataset_BUSI_with_GT/*/*_mask.png")
images = [mask_images.replace("_mask", "") for mask_images in masks]
dataset = pd.DataFrame(list(zip(images, masks)), columns=['image_path', 'mask_path'])
train, test= train_test_split(dataset, test_size=0.2, random_state=30)

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

model_names = ["unet", "unet++", "manet", "linknet", "fpn", "pspnet", "pan", "deeplabv3", "deeplabv3+"]


for i in [2, 3, 10, 20, 55, 67, 87, 96, 110, 130, 150]:
    predictions = []
    titles = []
    image = test_dataset[i][0]
    mask = test_dataset[i][1]
    image = image.to(device)
    for model_name in model_names:
        model = get_model(model_name).to(device)

        checkpoint_path = f'./log/{model_name}_best.pth'
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval() 
        
        with torch.no_grad(): 
            image_tensor = image.to(device).unsqueeze(0)  
            pred = model(image_tensor)
            pred = torch.sigmoid(pred)  
            pred = pred.squeeze().cpu()  
        
        predictions.append(pred.numpy())  
        titles.append(model_name)  

    plot_predictions(image.cpu().numpy(), mask.cpu().numpy(), predictions, titles, 0.5, image_path, i)
