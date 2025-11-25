"""
TP2 – Analyse des données (IFT599 / IFT799)

**Auteurs :**  
-  Ana Karen Lopez Carbajal (lopa2603)
-  Étienne Chaput (chae3018)
-  Anthony Glaude (glaa3301)

**Date de remise :** 25 novembre 2025  
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

class AETrainer:
    def __init__(self, model, train_loader, device,
                 epochs=50, learning_rate=1e-3, save_dir=None,noise_std=0.0):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.epochs = epochs
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.noise_std = noise_std

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            self.save_path = os.path.join(save_dir, f"{model.name}.pt")
        else:
            self.save_path = None

    def train(self):
        train_losses = []
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss = 0.0
            progress_bar = tqdm(self.train_loader, desc=f"training (epoch {epoch})", leave=True)
            counter = 1
            for batch in progress_bar:
                inputs = batch.to(self.device)
                if self.noise_std > 0:  # Cas d'analyse avec bruit (DAE)
                    noisy_inputs = inputs + self.noise_std * torch.randn_like(inputs)
                else:
                    noisy_inputs = inputs
                self.optimizer.zero_grad()
                outputs = self.model(noisy_inputs)
                loss = self.criterion(outputs, inputs)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                avg_loss = running_loss / counter
                progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
                counter += 1

            train_losses.append(avg_loss)

        if self.save_path is not None:
            torch.save(self.model.state_dict(), self.save_path)
            print(f"model saved to {self.save_path}")

        return train_losses

    @torch.no_grad()
    def compute_reconstruction_errors(self, data_loader):
        self.model.eval()
        errors = []
        for batch in data_loader:
            inputs = batch.to(self.device)
            outputs = self.model(inputs)
            batch_errors = torch.mean((outputs - inputs) ** 2, dim=1)
            errors.extend(batch_errors.cpu().numpy())
        return np.array(errors)
