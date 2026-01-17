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

    def add_lr_value(self,value):
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
        def rescale(values):
            return values
            #values = values / (values.std() + 1e-10)
            #values = values - values.min() * 2
            #return values
        for i,key in enumerate(self.train_log.keys()):
            train_val = rescale(np.array(self.train_log[key]))
            ax1.plot(
                np.arange(len(train_val)),
                train_val,
                color = colors[i],
                linestyle = 'solid',
                label = key,
            )
            if key in self.valid_log.keys():
                valid_val = rescale(np.array(self.valid_log[key]))
                ax1.plot(
                    np.arange(len(valid_val)),
                    valid_val,
                    color = colors[i],
                    linestyle = 'dashed',
                )
        ax2.plot(
            np.arange(len(self.lr)),
            self.lr,
            color = 'black',
            label = 'LR',
        )
        ax1.legend()
        ax1.set_xlabel('Epoch',fontsize=16)
        ax1.set_ylabel('Loss',fontsize=16)
        ax2.set_ylabel('Learning rate',fontsize=16)
        #ax1.set_yscale('log')
        ax2.set_yscale('log')

        fig.savefig(path)
        print (f"Loss curves saved as {path}")


