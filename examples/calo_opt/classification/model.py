from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import ClassificationDataset


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Pre-activation block
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding="same"),
        )

        # Skip projection if channels change
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

        self.final_relu = nn.ReLU()

    def forward(self, x):
        out = self.block(x)
        skip = self.skip(x)
        return self.final_relu(out + skip)


class CNNBlocks(nn.Module):
    def __init__(self, channels, dense_net=False, dropout=0.):
        super().__init__()
        self.dense_net = dense_net

        self.block1 = ResidualBlock(channels, channels*2, kernel_size=3)
        self.block2 = ResidualBlock(channels*2, channels*4, kernel_size=5)
        self.block3 = ResidualBlock(channels*4, channels*8, kernel_size=7)

        self.dropout = nn.Dropout(p=dropout)
        self.aggr = nn.AdaptiveAvgPool2d((1,1))

    @property
    def out_channels(self):
        if self.dense_net:
            return self.block1.out_channels + self.block2.out_channels + self.block3.out_channels
        else:
            return self.block3.out_channels

    def forward(self, x):
        h1 = self.block1(x)
        h2 = self.block2(h1)
        h3 = self.block3(h2)
        if self.dense_net:
            x = torch.cat([h1,h2,h3],axis=1)
        else:
            x = h3
        x = self.dropout(x)
        x = self.aggr(x).squeeze()
        return x

class Classification(nn.Module):
    def __init__(
        self,
        num_parameters: int,
        num_input_features: int,
        num_target_features: int,
        num_context_features: int,
        initial_means: List[np.float32],
        initial_stds: List[np.float32],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the shape of the model.

        Args:
            num_parameters (int): The number of detector parameters (length of the list of simulation parameters).
            num_input_features (int): The number of input features (sensors and other simulation outputs).
            num_target_features (int): The number of target features (quantities which are representatives
                of the detector's capabilities).
        """
        super().__init__()

        self.n_parameters = num_parameters
        self.n_input_features = num_input_features
        self.n_target_features = num_target_features
        self.n_context_features = num_context_features
        self.means = initial_means
        self.stds = initial_stds

        self.param_layers = nn.Sequential(
            nn.Linear(self.n_parameters + self.n_context_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        out_dim = 32

        self.input_layers = []
        for input_features in self.n_input_features:
            if len(input_features) == 3:
                blocks = CNNBlocks(input_features[0],dense_net=False)
                self.input_layers.append(blocks)
                out_dim += blocks.out_channels
            elif len(input_features) == 1:
                self.input_layers.append(
                    nn.Sequential(
                        nn.Linear(input_features[0],32),
                        nn.BatchNorm1d(32),
                        nn.ReLU(),
                        nn.Linear(32,32),
                        nn.BatchNorm1d(32),
                        nn.ReLU(),
                        nn.Linear(32,16),
                    )
                )
                out_dim += 16
            else:
                raise NotImplementedError
        self.input_layers = nn.ModuleList(self.input_layers)

        self.final_layers = nn.Sequential(
            nn.Linear(out_dim, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100,100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, num_target_features),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=1e-3)
        self.device = torch.device(device)


    def forward(self, parameters, x, c) -> torch.Tensor:
        """ Concatenate the detector parameters and the input
        """
        assert len(x) == len(self.input_layers)
        x = torch.cat(
            [
                layer(ix)
                for ix, layer in zip(x,self.input_layers)
            ],
            dim = 1,
        )
        pc = self.param_layers(torch.cat([parameters, c], dim=1))
        x = torch.cat([pc, x], dim=1)
        return self.final_layers(x)

    @staticmethod
    def loss(y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        assert y_pred.shape == y.shape, f'y has shape {y.shape}, but y_pred has shape {y_pred.shape}'
        loss = nn.BCEWithLogitsLoss()(y_pred,y)
        if loss.dim() == 2 and loss.shape[1] > 1:
            loss = loss.mean(dim=-1)
        return loss

    def train_model(
        self,
        train_dataset: ClassificationDataset,
        valid_dataset: ClassificationDataset,
        batch_size: int,
        n_epochs: int,
        lr: float,
        plotter = None,
        early_stopping = None,
    ):
        if early_stopping.early_stop:
            return
        print(f"Classification Training: {lr=}, {batch_size=}")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size*10, shuffle=False)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.to(self.device)

        for epoch in range(n_epochs):
            # Training #
            self.train()
            train_losses = torch.zeros(len(train_loader))
            for batch_idx, (detector_parameters, x, c, y) in enumerate(train_loader):
                detector_parameters: torch.Tensor = detector_parameters.to(self.device)
                x: List[torch.Tensor] = [ix.to(self.device) for ix in x]
                c: torch.Tensor = c.to(self.device)
                y: torch.Tensor = y.to(self.device)
                y_pred: torch.Tensor = self(detector_parameters, x, c)
                loss_per_event = self.loss(y,y_pred)
                loss = loss_per_event.clone().mean()
                train_losses[batch_idx] = loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # Validation #
            self.eval()
            valid_losses = torch.zeros(len(valid_loader))
            for batch_idx, (detector_parameters, x, c, y) in enumerate(valid_loader):
                detector_parameters: torch.Tensor = detector_parameters.to(self.device)
                x: List[torch.Tensor] = [ix.to(self.device) for ix in x]
                c: torch.Tensor = c.to(self.device)
                y: torch.Tensor = y.to(self.device)
                y_pred: torch.Tensor = self(detector_parameters, x, c)
                loss_per_event = self.loss(y,y_pred)
                loss = loss_per_event.clone().mean()
                valid_losses[batch_idx] = loss.item()

            print(f"Class Epoch: {epoch:4d} - Loss: {train_losses.mean():8.3f} - Val loss {valid_losses.mean():8.3f}")
            if plotter is not None:
                plotter.add_value('Loss (training)',train_losses.mean())
                plotter.add_value('Loss (validation)',valid_losses.mean())
                plotter.add_value('lr',lr)
            early_stopping(valid_losses.mean(),self)
            if early_stopping.early_stop:
                print ('Early stopping')
                break

        self.eval()

    def apply_model_in_batches(
        self,
        dataset: ClassificationDataset,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """ Apply the model in batches this is necessary because the model is too large to apply it to the
        whole dataset at once. The model is applied to the dataset in batches and the results are concatenated
        (the batch size is a hyperparameter).
        """
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        results = torch.zeros((len(dataset),self.n_target_features))
        loss_array = torch.zeros(len(dataset))
        mean_loss = 0.

        self.to(self.device)
        self.eval()

        for batch_idx, (detector_parameters, x, c, y) in enumerate(data_loader):
            detector_parameters = detector_parameters.to(self.device)
            x: List[torch.Tensor] = [ix.to(self.device) for ix in x]
            c: torch.Tensor = c.to(self.device)
            y: torch.Tensor = y.to(self.device)
            y_pred: torch.Tensor = self(detector_parameters, x, c)

            loss_per_event = self.loss(y,y_pred)
            loss = loss_per_event.clone().mean()
            mean_loss += loss.item()

            results[batch_idx * batch_size: (batch_idx + 1) * batch_size] = y_pred
            loss_array[batch_idx * batch_size: (batch_idx + 1) * batch_size] = loss_per_event.flatten()

        mean_loss /= len(data_loader)
        results = results.detach().cpu().numpy()
        loss_array = loss_array.detach().cpu().numpy()
        return results, loss_array, mean_loss
