import torch
import os
import wandb
import torch.nn.functional as F
from tqdm import tqdm

class Trainer:
    def __init__(self, model, model_name, num_epochs, optimizer, criterion, device, project_name):
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.model = model
        self.device = device
        self.log_interval = 500

        # Initialize wandb
        wandb.init(project=project_name, entity="weilunwu", name=model_name) 
        # wandb.watch(self.model, log_freq=500)

        self.train_losses = []
        self.val_losses = []
        self.train_dices = []
        self.val_dices = []

        self.best_model = None
        self.best_dice = 0.0
        self.best_epoch = 0

    def dice_coeff(self, predicted, target, smooth=1e-5):
        intersection = torch.sum(predicted * target)
        union = torch.sum(predicted) + torch.sum(target)
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice

    def save_best_model(self, epoch, dice):
        if dice > self.best_dice and dice > 0.65:
            self.best_dice = dice
            self.best_epoch = epoch
            self.best_model = self.model.state_dict()

            log_directory = 'log'
            os.makedirs(log_directory, exist_ok=True)
            filename = f'{log_directory}/{self.model_name}_epoch{epoch}_dice{dice:.4f}.pth'
            torch.save(self.best_model, filename)
            # wandb.save(filename) # Save the model file to wandb

    def train(self, train_loader, val_loader):
        for epoch in range(self.num_epochs):
            train_loss = 0.0
            val_loss = 0.0
            train_dice = 0.0
            val_dice = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")
            for images, masks in progress_bar:
                images, masks = images.to(self.device), masks.to(self.device)

                self.model.train()
                self.optimizer.zero_grad()

                outputs = self.model(images)
                outputs = F.sigmoid(outputs)
                loss = self.criterion(outputs, masks)
                dice = self.dice_coeff(outputs, masks)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_dice += dice.item()

                # if (i + 1) % self.log_interval == 0:
                #     wandb.log({"Train Loss": loss.item(), "Train Dice": dice.item()})

            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(self.device), masks.to(self.device)
                    outputs = self.model(images)
                    outputs = F.sigmoid(outputs)
                    loss = self.criterion(outputs, masks)
                    dice = self.dice_coeff(outputs, masks)

                    val_loss += loss.item()
                    val_dice += dice.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_train_dice = train_dice / len(train_loader)
            avg_val_dice = val_dice / len(val_loader)

            # Log epoch-level metrics
            wandb.log({
                "Avg Train Loss": avg_train_loss,
                "Avg Val Loss": avg_val_loss,
                "Avg Train Dice": avg_train_dice,
                "Avg Val Dice": avg_val_dice
            })

            self.save_best_model(epoch + 1, avg_val_dice)

    def get_metrics(self):
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_dices': self.train_dices,
            'val_dices': self.val_dices,
            'best_model': self.best_model,
            'best_dice': self.best_dice,
            'best_epoch': self.best_epoch
        }
