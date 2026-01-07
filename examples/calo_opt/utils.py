import copy
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

class LossPlotting:
    def __init__(self):
        self.train_log = {}
        self.valid_log = {}
        self.lr = []

    def add_lr_value(self,key,value):
        self.lr.append(value)

    def add_train_value(self,key,value):
        if key not in self.train_log.keys():
            self.train_log[key] = [value]
        else:
            self.train_log[key].append(value)

    def add_valid_value(self,key,value):
        if key not in self.valid_log.keys():
            self.valid_log[key] = [value]
        else:
            self.valid_log[key].append(value)

    def plot(self,path):
        fig,ax1 = plt.subplots(figsize=(6,5))
        ax2 = ax1.twinx()
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.train_log)))
        for i,key in enumerate(self.train_log.keys()):
            ax1.plot(
                np.arange(len(self.train_log[key])),
                self.train_log[key],
                color = colors[i],
                linestyle = 'solid',
                label = key,
            )
            if key in self.valid_log.keys():
                ax1.plot(
                    np.arange(len(self.valid_log[key])),
                    self.valid_log[key],
                    color = colors[i],
                    linestyle = 'dashed',
                )
        ax2.plot(
            np.arange(len(self.lr)),
            self.lr,
        )
        ax1.legend()
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax2.set_ylabel('Learning rate')
        ax1.set_yscale('log')
        ax2.set_yscale('log')

        fig.savefig(path)
        print (f"Loss curves saved as {path}")


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        if mode == 'min':
            self.best_score = np.inf
        elif mode == 'max':
            self.best_score = -np.inf
        else:
            raise ValueError("mode must be 'min' or 'max'")

        self.counter = 0
        self.epoch = 0
        self.saved_epoch = None
        self.early_stop = False
        self.best_state_dict = None

    def __call__(self, value, model=None):
        """
        Call this after each validation step.
        Args:
            value: The metric being monitored (e.g. val_loss).
            model: Pass the model if you want checkpoints saved.
        """
        self.epoch += 1

        improvement = (
            (self.mode == 'min' and value < self.best_score - self.min_delta) or
            (self.mode == 'max' and value > self.best_score + self.min_delta)
        )

        if improvement:
            self.best_score = value
            self.counter = 0
            if model is not None:
                self.saved_epoch = self.epoch
                self.best_state_dict = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_best_weights(self, model):
        """Call after training to load the best model parameters."""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            print (f'Restored model from epoch {self.saved_epoch}')
        else:
            print("Warning: No best_state_dict has been saved.")
