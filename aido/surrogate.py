import math
from typing import List, Tuple, Self
from tqdm.auto import tqdm
from copy import deepcopy
import random
import numpy as np
from scipy.optimize import linear_sum_assignment

import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.utils import to_dense_batch
from torchdiffeq import odeint
from torch.func import functional_call, vjp

import zuko

from aido.logger import logger
from aido.losses import HungarianMatching
from aido.utils import LossPlotting

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

class SurrogateDataset(Dataset):
    def __init__(
        self,
        feature_dict,
        max_seq_len,
        true_part,
        true_mask,
        reco_part,
        reco_mask,
        params,
        reco_loss = None,
        reco_time = None,
        samp_time = None,
        samp_part = None,
        samp_mask = None,
        random_reco = False,
        multiplicity = None,
    ):

        self.feature_dict = feature_dict
        self.random_reco = random_reco
        self.max_seq_len = max_seq_len
        self.multiplicity = multiplicity

        self.true_part = true_part
        self.true_mask = true_mask
        self.reco_part = reco_part
        self.reco_mask = reco_mask
        self.params = params
        self.reco_loss = reco_loss
        self.reco_time = reco_time
        self.samp_time = samp_time
        self.samp_part = samp_part
        self.samp_mask = samp_mask

        # Safety checks #
        assert self.true_part.shape[0] == self.reco_part.shape[0]
        assert self.true_part.shape[0] == self.true_mask.shape[0]
        assert self.reco_part.shape[0] == self.reco_mask.shape[0]
        assert self.true_part.shape[0] == self.params.shape[0]
        if self.reco_loss is not None:
            assert isinstance(self.reco_loss,dict)
            for key,losses in self.reco_loss.items():
                assert losses.shape[0] == self.true_part.shape[0]
        if self.reco_time is not None:
            assert self.reco_time.shape[0] == self.true_part.shape[0]
        if self.samp_time is not None:
            assert self.samp_time.shape[1] == self.true_part.shape[0]
        if self.samp_part is not None and self.samp_mask is not None:
            assert self.samp_part.shape[1] == self.true_part.shape[0]
            assert self.samp_mask.shape[1] == self.true_mask.shape[0]

    @property
    def oversampling(self):
        if self.samp_part is None:
            raise RuntimeError
        return self.samp_part.shape[0]

    @staticmethod
    def densify(x,ptr):
        num_nodes_per_graph = ptr[1:] - ptr[:-1]
        batch = torch.repeat_interleave(
            torch.arange(len(num_nodes_per_graph)),
            num_nodes_per_graph,
        )
        y,m = to_dense_batch(x,batch,fill_value=0.)
        return y,m

    @classmethod
    def load_sparse(cls,config,filepath):
        # Load data #
        out = torch.load(
            filepath,
            weights_only = False,
            map_location = torch.device('cpu'),
        )
        data = out['data']
        slices = out['slices']
        assert "particles" in data.node_types
        assert "vertices" in data.node_types
        # Record prameters #
        params = data['global_params'].unsqueeze(dim=1)
        # Timing #
        if config.surrogate.timing:
            times = torch.log(data['time']['times'])
        else:
            times = None
        # Record all features #
        feature_dict = {}
        true_part = []
        true_mask = None
        reco_part = []
        reco_mask = None
        idx = 0
        for name in config.surrogate.classification + config.surrogate.regression:
            # Record true particles #
            assert name in data["particles"].keys()
            y,m = cls.densify(
                x = data["particles"][name],
                ptr = slices["particles"][name],
            )
            true_part.append(y)
            if true_mask is None:
                true_mask = m
            else:
                assert torch.equal(true_mask,m)
            # Record reco particles #
            assert name in data["vertices"].keys(), f'Could not find {name} feature in vertices'
            y,m = cls.densify(
                x = data["vertices"][name],
                ptr = slices["vertices"][name],
            )
            reco_part.append(y)
            if reco_mask is None:
                reco_mask = m
            else:
                assert torch.equal(reco_mask,m)
            # Record features
            feature_dict[name] = torch.arange(idx,idx+y.shape[-1])
            idx += y.shape[-1]

        # Make global particle tensor #
        true_part = torch.concatenate(true_part,dim=-1)
        reco_part = torch.concatenate(reco_part,dim=-1)

        # Max sequence length #
        if config.surrogate.max_seq_len is not None:
            if reco_part.shape[1] < config.surrogate.max_seq_len:
                print (f'{reco_part.shape[1]} particles but max_seq_len is {config.surrogate.max_seq_len}, expanding by {config.surrogate.max_seq_len - reco_part.shape[1]}')
                reco_part = torch.cat(
                    (
                        reco_part,
                        torch.zeros(
                            reco_part.shape[0],
                            config.surrogate.max_seq_len - reco_part.shape[1],
                            reco_part.shape[2],
                        ),
                    ),
                    dim = 1,
                )
                reco_mask = torch.cat(
                    (
                        reco_mask,
                        torch.full(
                            (
                                reco_mask.shape[0],
                                config.surrogate.max_seq_len - reco_mask.shape[1],
                            ),
                            fill_value = False,
                        ),
                    ),
                    dim = 1,
                )
            elif reco_part.shape[1] > config.surrogate.max_seq_len:
                print (f'{reco_part.shape[1]} particles but max_seq_len is {config.surrogate.max_seq_len}, removing {reco_part.shape[1] - config.surrogate.max_seq_len} columns so in total {reco_part[:,config.surrogate.max_seq_len:].sum()} particles')
                reco_part = reco_part[:,:config.surrogate.max_seq_len]
                reco_maskg = reco_part[:,:config.surrogate.max_seq_len]


        # Order reco particles #
        if config.surrogate.ordering != 'random':
            random_reco = False
            assert config.surrogate.ordering in feature_dict.keys(), f'Could not find {config.surrogate.ordering} in {feature_dict.keys()} and is not "random"'
            feature_idx = feature_dict[config.surrogate.ordering][0]
            reco_ordering = torch.arange(reco_part.shape[1]).unsqueeze(0).repeat_interleave(reco_part.shape[0],dim=0)
            for i in range(reco_part.shape[0]):
                order = reco_part[i,reco_mask[i,:]][:,feature_idx].argsort(descending=True)
                reco_ordering[i,:len(order)] = order
            reco_part = reco_part[
                torch.arange(reco_part.shape[0]).unsqueeze(1),
                reco_ordering,
            ]
            reco_mask = reco_mask[
                torch.arange(reco_mask.shape[0]).unsqueeze(1),
                reco_ordering,
            ]
        else:
            random_reco = True

        return cls(
            feature_dict = feature_dict,
            max_seq_len = config.surrogate.max_seq_len,
            true_part = true_part,
            true_mask = true_mask,
            reco_part = reco_part,
            reco_mask = reco_mask,
            reco_time = times,
            params = params,
            random_reco = random_reco,
            multiplicity = config.surrogate.multiplicity,
        )

    def save_dense(self,path):
        out = {
            'feature_dict' : self.feature_dict,
            'true_part' : self.true_part,
            'true_mask' : self.true_mask,
            'reco_part' : self.reco_part,
            'reco_mask' : self.reco_mask,
            'params' : self.params,
        }
        if self.reco_loss is not None:
            out['reco_loss'] = self.reco_loss
        if self.samp_part is not None and self.samp_mask is not None:
            out['samp_part'] = self.samp_part
            out['samp_mask'] = self.samp_mask
        if self.reco_time is not None:
            out['reco_time'] = self.reco_time
        if self.samp_time is not None:
            out['samp_time'] = self.samp_time
        torch.save(out,path)
        print (f'Saved dense dataset to {path}')

    @classmethod
    def load_dense(cls,path):
        return cls(**torch.load(path,weights_only=False))

    def __len__(self):
        return self.true_part.shape[0]

    @property
    def n_features(self):
        return self.true_part.shape[2]

    def __getitem__(self,idx):
        event = {
            'params' : self.params[idx],
            'true' : {
                'data' : self.true_part[idx],
                'mask' : self.true_mask[idx],
            },
        }
        if self.multiplicity == 'hard':
            event['mult'] = F.one_hot(self.reco_mask[idx].sum() - 1,self.max_seq_len)
        if not self.random_reco:
            event['reco'] = {
                'data' : self.reco_part[idx],
                'mask' : self.reco_mask[idx],
            }
        else:
            ordering = torch.arange(len(self.reco_mask[idx]))
            N = self.reco_mask[idx].sum()
            ordering[:N] = torch.randperm(N)
            event['reco'] = {
                'data' : self.reco_part[idx][ordering],
                'mask' : self.reco_mask[idx][ordering],
            }
        if self.reco_time is not None:
            i = torch.randint(0, self.reco_time.shape[1], (1,)).item()
            event['time'] = self.reco_time[idx,i:i+1]
        return event

    def get_shapes(self):
        shapes = {
            'true': self.true_part.shape[-1],
            'reco': self.reco_part.shape[-1],
            'params': self.params.shape[-1],
        }
        if self.multiplicity == 'hard':
            shapes['mult'] = self.max_seq_len
        if self.reco_time is not None:
            shapes['time'] = 1
        return shapes

    def get_input_mean(self,values,mask=None):
        means = torch.zeros((values.shape[-1],),dtype=values.dtype)
        for i in range(values.shape[-1]):
            x = values[...,i]
            if mask is not None:
                x = x[mask]
            if not ((x == 0) | (x == 1)).all():
                means[i] = x.mean()
        return means

    def get_input_std(self,values,mask=None):
        stds = torch.ones((values.shape[-1],),dtype=values.dtype)
        for i in range(values.shape[-1]):
            x = values[...,i]
            if mask is not None:
                x = x[mask]
            if not ((x == 0) | (x == 1)).all():
                stds[i] = x.std()
        return stds

    def get_means(self):
        means = {
            'true' : self.get_input_mean(self.true_part,self.true_mask),
            'reco' : self.get_input_mean(self.reco_part,self.reco_mask),
            'params' : self.get_input_mean(self.params,None)
        }
        if self.reco_time is not None:
            means['time'] = self.get_input_mean(self.reco_time.reshape(-1,1))
        return means


    def get_stds(self):
        stds = {
            'true' : self.get_input_std(self.true_part,self.true_mask),
            'reco' : self.get_input_std(self.reco_part,self.reco_mask),
            'params' : self.get_input_std(self.params,None)
        }
        if self.reco_time is not None:
            stds['time'] = self.get_input_std(self.reco_time.reshape(-1,1))
        return stds

    def plot(self,config,idx_event,idx_samp=0,savepath=None,show=False):
        # Safety checks #
        assert 'pos' in self.feature_dict.keys()

        # Make figure and recover event #
        ncols = 2
        if self.samp_part is not None:
            ncols += len(config.surrogate.classification) + len(config.surrogate.regression)
        fig, axs = plt.subplots(ncols=ncols,figsize=(5*ncols, 5))
        plt.subplots_adjust(left=0.10,right=0.9,bottom=0.1,top=0.90,wspace=0.3)

        true = self.true_part[idx_event][self.true_mask[idx_event]]
        reco = self.reco_part[idx_event][self.reco_mask[idx_event]]
        print (self.feature_dict)
        print ('true')
        print (true)
        print ('reco')
        print (reco)
        if self.samp_part is not None:
            samp = self.samp_part[idx_samp,idx_event][self.samp_mask[idx_samp,idx_event]]
            print ('samp')
            print (samp)
        else:
            samp = None

        # Matching #
        loss_matching = HungarianMatching(
            feature_dict = self.feature_dict,
            loss_factors = config.loss.loss_factors,
            fake_penalty = config.loss.fake_penalty,
            missing_penalty = config.loss.missing_penalty,
            classification = config.surrogate.classification,
            regression = config.surrogate.regression,
        )
        cost_matrix_true_reco = loss_matching.pairwise_cost(true,reco)
        true_idx,reco_idx = linear_sum_assignment(loss_matching.pairwise_cost(true,reco))
        groups = [
            [a,b]
            for a,b in zip(true_idx,reco_idx)
        ]
        for a in range(true.shape[0]):
            if a not in true_idx:
                groups.append([a,None])
        for b in range(reco.shape[0]):
            if b not in reco_idx:
                groups.append([None,b])
        if samp is not None:
            reco_idx,samp_idx = linear_sum_assignment(loss_matching.pairwise_cost(reco,samp))
            for i in range(len(groups)):
                if groups[i][1] in reco_idx:
                    groups[i].append(samp_idx[np.where(reco_idx==groups[i][1])[0][0]])
                else:
                    groups[i].append(None)
            for c in range(samp.shape[0]):
                if c not in samp_idx:
                    groups.append([None,None,c])

        # Position plot #
        cmap = plt.get_cmap("tab20b")
        colors = list(cmap.colors)
        random.shuffle(colors)
        assert len(colors) >= len(groups)

        def format_info(array):
            s = ''
            if 'E' in self.feature_dict.keys():
                s += f'E = {float(array[self.feature_dict["E"]]):3.2f}\n'
            if 'id' in self.feature_dict.keys():
                val_id = array[self.feature_dict["id"]]
                s += f'ID : {val_id.argmax():d} ({val_id.max():+.2f})'
            return s

        xs = [.05,.40,.75]
        y = 0.9
        extra_markers = ['o', 's', '^', 'D', 'P', 'v']

        base_box = dict(
            y = y + 0.05,
            transform = axs[1].transAxes,
            va = "center",
            ha = "center",
            fontsize = 14,
            family = "monospace",
            color = 'black',
            bbox = dict(
                facecolor = "none",
                edgecolor = "none",
            ),
        )
        axs[1].text(
            x = xs[0]+0.075,
            s = 'True',
            **base_box,
        )
        axs[1].text(
            x = xs[1]+0.075,
            s = 'Reco',
            **base_box,
        )
        axs[1].text(
            x = xs[2]+0.075,
            s = 'Surrogate',
            **base_box,
        )


        min_pos = +math.inf
        max_pos = -math.inf
        for i in range(len(groups)):
            # Decide font color #
            r,g,b = colors[i]
            luminance = 0.2126*r + 0.7152*g + 0.0722*b
            if luminance < 0.5:
                font_color = "white"
            else:
                font_color = "black"
            # Plot position #
            idx_pos = self.feature_dict['pos']
            a,b = groups[i][:2]
            base_box = dict(
                y = y,
                transform = axs[1].transAxes,
                va = "top",
                ha = "left",
                fontsize = 6,
                family = "monospace",
                color = font_color,
                bbox = dict(
                    boxstyle = "round,pad=0.8",
                    facecolor = colors[i],
                    edgecolor = "none",
                ),
            )
            height = 0.
            if a is not None:
                axs[0].scatter(
                    true[a,idx_pos[0]],
                    true[a,idx_pos[1]],
                    marker = 'X',
                    s = 50,
                    edgecolor = 'black',
                    facecolor = colors[i],
                )
                box_txt = format_info(true[a])
                height = max(height,box_txt.count('\n')*0.03)
                axs[1].text(
                    x = xs[0],
                    s = box_txt,
                    **base_box,
                )
                min_pos = min(min_pos,true[a,idx_pos].min())
                max_pos = max(max_pos,true[a,idx_pos].max())
            if b is not None:
                axs[0].scatter(
                    reco[b,idx_pos[0]],
                    reco[b,idx_pos[1]],
                    marker = 'o',
                    s = 50,
                    edgecolor = 'black',
                    facecolor = colors[i],
                )
                box_txt = format_info(reco[b])
                height = max(height,box_txt.count('\n')*0.03)
                axs[1].text(
                    x = xs[1],
                    s = box_txt,
                    **base_box,
                )
                min_pos = min(min_pos,reco[b,idx_pos].min())
                max_pos = max(max_pos,reco[b,idx_pos].max())
            if samp is not None and groups[i][2] is not None:
                c = groups[i][2]
                axs[0].scatter(
                    samp[c,idx_pos[0]],
                    samp[c,idx_pos[1]],
                    marker = 'v',
                    s = 50,
                    edgecolor = 'black',
                    facecolor = colors[i],
                )
                box_txt = format_info(samp[c])
                height = max(height,box_txt.count('\n')*0.03)
                axs[1].text(
                    x = xs[2],
                    s = box_txt,
                    **base_box,
                )
                min_pos = min(min_pos,samp[c,idx_pos].min())
                max_pos = max(max_pos,samp[c,idx_pos].max())
            y -= (height + 0.075)

            # Other plots #
            if samp is not None and b is not None and c is not None:
                for j, feature in enumerate(config.surrogate.regression + config.surrogate.classification,2):
                    idx_feat = self.feature_dict[feature]
                    assert len(extra_markers) >= len(idx_feat)
                    for l,k in enumerate(idx_feat):
                        axs[j].scatter(
                            reco[b,k],
                            samp[c,k],
                            marker = extra_markers[l],
                            facecolor = colors[i],
                            edgecolor = 'black',
                        )


        # Dummy plot for legend #
        axs[0].scatter(
            [],[],
            marker = 'X',
            s = 50,
            edgecolor = 'black',
            facecolor = 'white',
            label = 'True',
        )
        axs[0].scatter(
            [],[],
            marker = 'o',
            s = 50,
            edgecolor = 'black',
            facecolor = 'white',
            label = 'Reco',
        )
        if samp is not None:
            axs[0].scatter(
                [],[],
                marker = 'v',
                s = 50,
                edgecolor = 'black',
                facecolor = 'white',
                label = 'Surrogate',
            )

        # Esthetics #
        axs[0].legend(frameon=False,fontsize=14)
        axs[0].set_xlim(min_pos-5,max_pos+5)
        axs[0].set_ylim(min_pos-5,max_pos+5)
        axs[0].set_xlabel('x [cm]',fontsize=16)
        axs[0].set_ylabel('y [cm]',fontsize=16)
        axs[0].set_aspect("equal", adjustable="box")
        axs[1].axis('off')

        if samp is not None:
            for j, feature in enumerate(config.surrogate.regression + config.surrogate.classification,2):
                idx_feat = self.feature_dict[feature]
                axs[j].set_aspect("equal", adjustable="box")
                axs[j].set_xlabel(f'Reco {feature}',fontsize=16)
                axs[j].set_ylabel(f'Surrogate {feature}',fontsize=16)
                min_val = min([reco[:,idx_feat].min(),samp[:,idx_feat].min()])
                if min_val < 0:
                    min_val *= 1.2
                else:
                    min_val *= 0.8
                max_val = max([reco[:,idx_feat].max(),samp[:,idx_feat].max()])
                if max_val < 0:
                    max_val *= 0.8
                else:
                    max_val *= 1.2
                axs[j].set_xlim(min_val,max_val)
                axs[j].set_ylim(min_val,max_val)
                axs[j].plot(
                    [min_val,max_val],
                    [min_val,max_val],
                    color = 'grey',
                    linewidth = 2,
                    linestyle = '--',
                )
                for k in range(len(idx_feat)):
                    axs[j].scatter(
                        [],[],
                        marker = extra_markers[k],
                        s = 50,
                        label = f'Dim {k}',
                        edgecolor = 'black',
                        facecolor = 'white',
                    )
                axs[j].legend(frameon=False,fontsize=14)


        if savepath is not None:
            fig.savefig(savepath)
        if show:
            plt.show()
        plt.close()

    def plot_sampling(self,config,idx,savepath=None,show=False):
        # Safety checks #
        assert 'pos' in self.feature_dict.keys()
        assert self.samp_part is not None

        # Figrue #
        ncols = len(config.surrogate.classification) + len(config.surrogate.regression)
        fig, axs = plt.subplots(ncols=ncols,figsize=(6*ncols, 5))
        if not isinstance(axs,np.ndarray):
            axs = np.array([axs])
        plt.subplots_adjust(left=0.10,right=0.9,bottom=0.1,top=0.90,wspace=0.3)

        true = self.true_part[idx][self.true_mask[idx]]
        reco = self.reco_part[idx][self.reco_mask[idx]]
        samps = [
            self.samp_part[i,idx][self.samp_mask[i,idx]]
            for i in range(self.samp_part.shape[0])
        ]

        # Matching #
        loss_matching = HungarianMatching(
            feature_dict = self.feature_dict,
            loss_factors = config.loss.loss_factors,
            fake_penalty = config.loss.fake_penalty,
            missing_penalty = config.loss.missing_penalty,
            classification = config.surrogate.classification,
            regression = config.surrogate.regression,
        )
        cost_matrix_true_reco = loss_matching.pairwise_cost(true,reco)
        true_idx,reco_idx = linear_sum_assignment(loss_matching.pairwise_cost(true,reco))

        samp_idxs = [
            linear_sum_assignment(loss_matching.pairwise_cost(true,samps[i]))[1]
            for i in range(len(samps))
        ]

        # Position plot #
        cmap = plt.get_cmap("tab20b")
        colors = list(cmap.colors)
        random.shuffle(colors)

        idx_pos = self.feature_dict['pos']
        min_pos = min(
            [reco[:,idx_pos].min()] + [
                samp[:,idx_pos].min()
                for samp in samps
            ]
        )
        max_pos = max(
            [reco[:,idx_pos].max()] + [
                samp[:,idx_pos].max()
                for samp in samps
            ]
        )

        groups = [
            [a,b]
            for a,b in zip(true_idx,reco_idx)
        ]
        for a in range(true.shape[0]):
            if a not in true_idx:
                groups.append([a,None])
        for b in range(reco.shape[0]):
            if b not in reco_idx:
                groups.append([None,b])
        groups.append([None,None])
        for samp in samps:
            true_idx, samp_idx = linear_sum_assignment(loss_matching.pairwise_cost(true,samp))
            for i in range(len(groups)):
                if groups[i][0] in true_idx:
                    groups[i].append(samp_idx[np.where(true_idx==groups[i][0])[0][0]])
                else:
                    groups[i].append(None)
            for c in range(samp_idx.shape[0]):
                if c not in samp_idx:
                    groups[-1].append(c)
                else:
                    groups[-1].append(None)




        for i in range(len(groups)):
            a,b = groups[i][:2]
            cs = groups[i][2:]
            for c,samp in zip(cs,samps):
                if c is not None:
                    axs[0].scatter(
                            samp[c,idx_pos[0]],
                            samp[c,idx_pos[1]],
                            marker = '.',
                            s = 100,
                            alpha = 0.5,
                            edgecolor = None,
                            facecolor = colors[i],
                        )
            if a is not None:
                axs[0].scatter(
                        true[a,idx_pos[0]],
                        true[a,idx_pos[1]],
                        marker = 'X',
                        s = 50,
                        edgecolor = 'black',
                        facecolor = colors[i],
                    )
            if b is not None:
                axs[0].scatter(
                        reco[b,idx_pos[0]],
                        reco[b,idx_pos[1]],
                        marker = 'o',
                        s = 50,
                        edgecolor = 'black',
                        facecolor = colors[i],
                    )
        axs[0].scatter(
            [],[],
            marker = 'X',
            s = 50,
            edgecolor = 'black',
            facecolor = 'white',
            label = 'True',
        )
        axs[0].scatter(
            [],[],
            marker = 'o',
            s = 50,
            edgecolor = 'black',
            facecolor = 'white',
            label = 'Reco',
        )
        if samp is not None:
            axs[0].scatter(
                [],[],
                marker = '.',
                s = 100,
                alpha = 0.5,
                edgecolor = 'black',
                facecolor = 'black',
                label = 'Surrogate',
            )

        axs[0].legend(frameon=False,fontsize=14)
        axs[0].set_xlim(min_pos-5,max_pos+5)
        axs[0].set_ylim(min_pos-5,max_pos+5)
        axs[0].set_xlabel('x [cm]',fontsize=16)
        axs[0].set_ylabel('y [cm]',fontsize=16)
        axs[0].set_aspect("equal", adjustable="box")

        features = [
            feature
            for feature in config.surrogate.regression + config.surrogate.classification
            if feature != 'pos'
        ]
        for j, feature in enumerate(features,1):
            idx_feat = self.feature_dict[feature]
            min_val = min(
                [reco[:,idx_feat].min()] + [
                    samp[:,idx_feat].min()
                    for samp in samps
                ]
            )
            if min_val < 0:
                min_val *= 1.2
            else:
                min_val *= 0.8
            max_val = max(
                [reco[:,idx_feat].max()] + [
                    samp[:,idx_feat].max()
                    for samp in samps
                ]
            )
            if max_val < 0:
                max_val *= 0.8
            else:
                max_val *= 1.2

            for k,idx in enumerate(idx_feat):
                data = []
                positions = []
                violin_colors = []
                for i in range(len(groups)):
                    b = groups[i][1]
                    if b is None:
                        continue
                    cs = groups[i][2:]
                    if all([c is None for c in cs]):
                        continue
                    positions.append(float(reco[b,idx]))
                    data.append(
                        [
                            float(samp[c,idx])
                            for c,samp in zip(cs,samps)
                            if c is not None
                        ]
                    )
                    violin_colors.append(colors[i])
                parts = axs[j].violinplot(
                    dataset = data,
                    positions = positions,
                    vert = True,
                    showmeans = False,
                    showmedians = True,
                    showextrema = False,
                    widths = float(max_val-min_val)/10,
                    #label = f'Dim {k}',
                )
                for body, color in zip(parts['bodies'], violin_colors):
                    body.set_facecolor(color)
                    body.set_edgecolor("black")
                    body.set_alpha(0.5)

            axs[j].plot(
                [min_val,max_val],
                [min_val,max_val],
                color = 'grey',
                linewidth = 2,
                linestyle = '--',
            )
            axs[j].set_xlim(min_val,max_val)
            axs[j].set_ylim(min_val,max_val)
            axs[j].set_aspect("equal", adjustable="box")
            axs[j].set_xlabel(f'Reco {feature}',fontsize=16)
            axs[j].set_ylabel(f'Surrogate {feature}',fontsize=16)
            #axs[j].legend(frameon=False,fontsize=14)




        if savepath is not None:
            fig.savefig(savepath)
        if show:
            plt.show()
        plt.close()




class Encoder(nn.Module):
    def __init__(
        self,
        dim_embed: int,
        nhead: int = 8,
        num_layers: int = 6,
        expansion_factor: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dim_embed = dim_embed

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.dim_embed,
            nhead = nhead,
            dim_feedforward = self.dim_embed * expansion_factor,
            dropout = dropout,
            batch_first = True,
            activation = 'gelu',
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers = num_layers,
            norm = nn.LayerNorm(self.dim_embed),
        )

    def forward(self, x, mask):
        x = self.encoder(
            x,
            src_key_padding_mask = mask,
        )
        return x

class PositionEmbedding(nn.Module):
    def __init__(self,dim_embed,max_seq_len,dropout=0.):
        super().__init__()

        self.dim_embed = dim_embed
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(
            num_embeddings = self.max_seq_len,
            embedding_dim = self.dim_embed,
        )
        self.register_buffer(
            "positions",
            torch.arange(self.max_seq_len, dtype=torch.long).unsqueeze(0)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        if x.shape[1] > self.max_seq_len:
            raise RuntimeError(f'Expected max {self.max_seq_len} elements in x, got {x.shape[1]}')
        positions = self.positions[:, :x.shape[1]]
        x = x + self.embedding(positions)
        return self.dropout(x)


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim_embed, max_seq_len, dropout=0.0):
        super().__init__()

        self.dim_embed = dim_embed
        self.max_seq_len = max_seq_len

        self.dropout = nn.Dropout(p=dropout)

        # Create sinusoidal positional encoding once
        pe = torch.zeros(max_seq_len, dim_embed)  # [seq_len, dim]

        position = torch.arange(0, max_seq_len).unsqueeze(1).float()  # [seq_len, 1]

        div_term = torch.exp(
            torch.arange(0, dim_embed, 2).float() * (-math.log(10000.0) / dim_embed)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        # Shape: [1, max_seq_len, dim_embed]
        pe = pe.unsqueeze(0)

        # Register as buffer (not trainable)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, seq_len, embedding_dim]
        """
        seq_len = x.size(1)

        if seq_len > self.max_seq_len:
            raise RuntimeError(
                f'Expected max {self.max_seq_len} elements in x, got {seq_len}'
            )

        # Add positional encoding (broadcast over batch)
        x = x + self.pe[:, :seq_len, :]

        return self.dropout(x)

class Decoder(nn.Module):
    def __init__(
        self,
        dim_embed: int,
        nhead: int = 8,
        num_layers: int = 6,
        expansion_factor: int = 4,
        dropout: float = 0.1,
        position_encoding = None,
        is_causal: bool = True,
    ):
        super().__init__()

        self.is_causal = is_causal

        decoder_layer = nn.TransformerDecoderLayer(
            d_model = dim_embed,
            nhead = nhead,
            dim_feedforward = dim_embed * expansion_factor,
            dropout = dropout,
            batch_first = True,
            activation = 'gelu',
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers = num_layers,
            norm = nn.LayerNorm(dim_embed),
        )
        self.position_encoding = position_encoding

    def generate_square_subsequent_mask(self,size,device):
        return torch.triu(
            torch.full((size, size), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(
        self,
        x,
        mask,
        memory,
        memory_mask,
    ):
        if self.position_encoding:
            x = self.position_encoding(x)
        x = self.decoder(
            x,
            memory,
            tgt_mask = self.generate_square_subsequent_mask(x.shape[1],x.device) if self.is_causal else None,
            tgt_key_padding_mask = mask,
            memory_key_padding_mask = memory_mask,
            tgt_is_causal = self.is_causal,
        )
        return x


class CrossEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_embed: int,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()

        # Cross-attention: Q = slots, K/V = encoder output
        self.cross_attn = nn.MultiheadAttention(
            embed_dim = dim_embed,
            num_heads = nhead,
            dropout = dropout,
            batch_first = True,
        )

        # Feed-forward network
        self.linear1 = nn.Linear(dim_embed, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim_embed)

        # Normalization
        self.norm1 = nn.LayerNorm(dim_embed)
        self.norm2 = nn.LayerNorm(dim_embed)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(
        self,
        slots: torch.Tensor,
        x : torch.Tensor,
        padding_mask : torch.Tensor | None = None,
        attn_mask : torch.Tensor | None = None,
    ):
        """
        Args:
            slots: (B, S, D)
            encoder_out: (B, N, D)
            encoder_padding_mask: (B, N), True for padded positions

        Returns:
            updated_slots: (B, S, D)
        """

        # --- Cross-attention ---
        attn_out, _ = self.cross_attn(
            query = slots,
            key = x,
            value = x,
            key_padding_mask = padding_mask,
            attn_mask = attn_mask,
            need_weights=False,
        )

        # Residual + norm
        slots = self.norm1(slots + self.dropout1(attn_out))

        # --- Feed-forward ---
        ff_out = self.linear2(
            self.dropout(self.activation(self.linear1(slots)))
        )

        # Residual + norm
        slots = self.norm2(slots + self.dropout2(ff_out))

        return slots



class CrossEncoder(nn.Module):
    def __init__(
        self,
        dim_embed: int,
        nhead: int = 8,
        num_layers: int = 6,
        expansion_factor: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            encoder_layer: a SlotCrossEncoderLayer instance
            num_layers: number of layers to stack
            norm: optional final LayerNorm
        """
        super().__init__()

        self.dim_embed = dim_embed

        self.token = nn.Parameter(
            torch.randn(1,1,self.dim_embed),
            requires_grad = True,
        )

        encoder_layer = CrossEncoderLayer(
            dim_embed = dim_embed,
            nhead = nhead,
            dim_feedforward = dim_embed * expansion_factor,
            dropout = dropout,
            activation = 'gelu',

        )
        self.layers = nn.ModuleList(
            [deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(dim_embed)

    def forward(
        self,
        x : torch.Tensor,
        mask : torch.Tensor | None = None,
    ):

        token = self.token.expand(x.shape[0], -1, -1)

        for layer in self.layers:
            token = layer(token,x,mask)

        if self.norm is not None:
            token = self.norm(token)

        return token.squeeze(dim=1)

class Memory(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_embed: int,
        nhead: int = 8,
        num_layers: int = 6,
        expansion_factor: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            encoder_layer: a SlotCrossEncoderLayer instance
            num_layers: number of layers to stack
            norm: optional final LayerNorm
        """
        super().__init__()

        self.dim_embed = dim_embed

        self.token = nn.Parameter(
            torch.randn(1,1,self.dim_embed),
            requires_grad = True,
        )

        self.embedding = nn.Linear(dim_in,self.dim_embed)

        encoder_layer = CrossEncoderLayer(
            dim_embed = dim_embed,
            nhead = nhead,
            dim_feedforward = dim_embed * expansion_factor,
            dropout = dropout,
            activation = 'gelu',

        )
        self.layers = nn.ModuleList(
            [deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(dim_embed)

    def forward(
        self,
        x : torch.Tensor,
        mask : torch.Tensor | None = None,
        is_causal: bool = False,
    ):
        B,N,_ = x.shape
        if is_causal:
            attn_mask = torch.triu(
                torch.full((N,N), float('-inf'), device=x.device),
                diagonal = 0,  # exclude diagonal — slot i cannot see itself
            )
        else:
            attn_mask = None

        token = self.token.expand(B,N,-1)

        x = self.embedding(x)

        for layer in self.layers:
            token = layer(token,x,mask,attn_mask)

        if self.norm is not None:
            token = self.norm(token)

        return token

    def summarise(self,x):
        token = self.token.expand(x.shape[0],-1,-1)
        if x.shape[1] == 0:
            x = torch.zeros(x.shape[0], 1, self.dim_embed, device=x.device)
        else:
            x = self.embedding(x)
        for layer in self.layers:
            token = layer(token, x)
        if self.norm is not None:
            token = self.norm(token)
        return token.squeeze(1)


class ClassifierMultiplicity(nn.Module):
    def __init__(
        self,
        embeddings,
        encoder,
        cross,
        max_seq_len,
        dropout,
    ):
        super().__init__()

        self.max_seq_len = max_seq_len

        self.embeddings = embeddings
        self.encoder = encoder
        self.cross = cross

        self.layers = nn.Sequential(
                nn.Linear(self.encoder.dim_embed,64),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(64,64),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(64,32),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(32,self.max_seq_len),
            )
        self.loss_function = nn.CrossEntropyLoss(reduction='none')

    def forward(self,batch):
        x = torch.cat(
            [
                self.embeddings['params'](
                    batch['params']
                ),
                self.embeddings['true'](
                    batch['true']['data']
                ),
            ],
            dim = 1,
        )
        m = torch.cat(
            [
                torch.full(
                    (batch['true']['mask'].shape[0],1),
                    fill_value = True,
                ).to(batch['true']['mask'].device),
                batch['true']['mask'],
            ],
            dim = 1,
        )
        y = self.encoder(x,~m)
        y = self.cross(x,~m)
        return self.layers(y)

    def loss(self,batch):
        y = self(batch)
        t = F.one_hot(batch['true']['mask'].sum(dim=1) - 1, self.max_seq_len).float()
        return self.loss_function(y,t)

    def sample(self,batch):
        probs = torch.softmax(self(batch), dim=-1)
        samples = torch.multinomial(probs, num_samples=1).squeeze(-1)
        idx = torch.arange(self.max_seq_len, device=samples.device)
        mask = idx.unsqueeze(0) <= samples.unsqueeze(1)
        return mask

class ClassifierExist(nn.Module):
    def __init__(self,dim_embed,alpha,gamma,max_seq_len,dropout):
        super().__init__()

        self.dim_embed = dim_embed

        self.layers = nn.Sequential(
                nn.Linear(self.dim_embed,64),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(64,64),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(64,32),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(32,1),
            )
        self.alpha = alpha
        self.gamma = gamma

    def focal_loss(self,y,t):
        bce = F.binary_cross_entropy_with_logits(
            y, t, reduction="none"
        )
        probs = torch.sigmoid(y)
        pt = probs * t + (1 - probs) * (1 - t)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce
        return loss

    def forward(self,x):
        return self.layers(x)

    def loss(self,y,mask):
        return self.focal_loss(y,mask.float()).mean(dim=-1)

    def sample(self,x):
        probs = torch.sigmoid(self(x)).squeeze(-1)
        mask = torch.bernoulli(probs).bool()
        return mask

class ClassifierHead(nn.Module):
    def __init__(self,dim_embed,dim_encoding,n_classes,dropout):
        super().__init__()
        self.dim_embed = dim_embed
        self.dim_encoding = dim_encoding
        self.n_classes = n_classes

        self.layers = nn.Sequential(
                nn.Linear(self.dim_embed,64),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(64,64),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(64,32),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(32,self.n_classes),
            )

        self.encodings = nn.Parameter(
            torch.randn(self.n_classes,self.dim_encoding),
            requires_grad = True,
        )

    def forward(self,x):
        return self.layers(x)

    def encode(self,logits):
        probs = F.softmax(logits,dim=-1)
        return torch.matmul(probs, self.encodings)

    def loss(self,y,m,t):
        # y are pred logits
        # m is mask
        # t are true logits
        loss = F.kl_div(
            input = F.log_softmax(y,dim=-1),
            target = F.log_softmax(t,dim=-1),
            log_target = True,
            reduction = 'none',
        ).sum(dim=-1) # sum across classes
        loss = (loss * m).sum(dim=-1)# / mask.sum(dim=-1)
        return loss

class Surrogate(nn.Module):
    def __init__(
        self,
        shapes,
        means,
        stds,
        max_seq_len,
        feature_dict,
        classification,
        regression,
        multiplicity,
        loss_factors,
        device,
    ):
        super().__init__()

        self.dim_embed = 64
        self.feature_dict = feature_dict
        assert set(self.feature_dict.keys()) == set(classification+regression), f'Mismatch beteen features in inputs ({list(self.feature_dict.keys())}) and their split into classification ({classification}) and regression ({regression})'
        self.multiplicity = multiplicity
        assert self.multiplicity in ['none','hard','soft']
        self.device = device

        # Parameters #
        self.means = nn.ParameterDict(
            {
                key : nn.Parameter(tensor,requires_grad=False)
                for key,tensor in means.items()
            }
        )
        self.stds = nn.ParameterDict(
            {
                key : nn.Parameter(tensor,requires_grad=False)
                for key,tensor in stds.items()
            }
        )
        self.loss_factors = nn.ParameterDict(
            {
                key : nn.Parameter(torch.tensor(val),requires_grad=False)
                for key,val in loss_factors.items()
            }
        )
        self.max_seq_len = max_seq_len

        # Pre-transformer transformations #
        self.embeddings = nn.ModuleDict(
            {
                name : nn.Sequential(
                    nn.Linear(shape,self.dim_embed),
                    nn.GELU(),
                    nn.Linear(self.dim_embed,self.dim_embed),
                )
                for name,shape in shapes.items()
            }
        )

        # Multiplicity #
        if self.multiplicity == 'hard':
            self.mult_hard = ClassifierMultiplicity(
                embeddings = nn.ModuleDict(
                    {
                        name : nn.Sequential(
                            nn.Linear(shapes[name],self.dim_embed),
                            nn.GELU(),
                            nn.Linear(self.dim_embed,self.dim_embed),
                        )
                        for name in ['true','params']
                    }
                ),
                encoder = Encoder(
                    dim_embed = self.dim_embed,
                    nhead = 4,
                    num_layers = 3,
                    expansion_factor = 4,
                    dropout = 0.1,
                ),
                cross = CrossEncoder(
                    dim_embed = self.dim_embed,
                    nhead = 4,
                    num_layers = 3,
                    expansion_factor = 4,
                    dropout = 0.1,
                ),
                max_seq_len = self.max_seq_len,
                dropout = 0.1,
            )
        #if self.multiplicity == 'soft':
        #    self.mult_soft_head = ClassifierExist(
        #        dim_embed = self.dim_embed,
        #        alpha = 0.25,
        #        gamma = 2,
        #        max_seq_len = self.max_seq_len,
        #        dropout = 0.1,
        #    )

        # Timing #
        if 'time' in shapes.keys():
            self.time_encoding = CrossEncoder(
                dim_embed = self.dim_embed,
                nhead = 4,
                num_layers = 3,
                expansion_factor = 2,
                dropout = 0.2,
            )
            self.time_head = zuko.flows.NSF(
                features = 1,
                context = self.dim_embed,
                bins = 16,
                transforms = 3,
                randperm = True,
                hidden_features = [128,128],
            )
        else:
            self.time_encoding = None
            self.time_head = None

        # Miscellanious #
        self.surrogate_loss = []
        self.order = torch.tensor(
            [
                idx
                for feature in regression + classification
                for idx in self.feature_dict[feature]
            ]
        )

    def set_to_device(self,device):
        self.device = device
        self.to(self.device)

    def apply_preprocessing(self, batch):
        # instead of replacing the original tensors
        x_true = (batch['true']['data'] - self.means['true']) / self.stds['true']
        x_reco = (batch['reco']['data'] - self.means['reco']) / self.stds['reco']
        x_params = (batch['params'] - self.means['params']) / self.stds['params']
        batch_new = {
            'true': {'data': x_true, 'mask': batch['true']['mask']},
            'reco': {'data': x_reco, 'mask': batch['reco']['mask']},
            'params': x_params,
        }
        if 'time' in batch.keys():
            batch_new['time'] = (batch['time'] - self.means['time']) / self.stds['time']
        if 'mult' in batch.keys():
            batch_new['mult'] = batch['mult']
        return batch_new

    def undo_preprocessing(self, batch):
        # instead of replacing the original tensors
        x_true = (batch['true']['data'] * self.stds['true']) + self.means['true']
        x_reco = (batch['reco']['data'] * self.stds['reco']) + self.means['reco']
        x_params = (batch['params'] * self.stds['params']) + self.means['params']
        batch_new = {
            'true': {'data': x_true, 'mask': batch['true']['mask']},
            'reco': {'data': x_reco, 'mask': batch['reco']['mask']},
            'params': x_params,
        }
        if 'time' in batch.keys():
            batch_new['time'] = (batch['time'] * self.stds['time']) + self.means['time']
        if 'mult' in batch.keys():
            batch_new['mult'] = batch['mult']
        return batch_new

    @staticmethod
    def move_batch(batch,device):
        batch_new = {
            'true': {
                'data': batch['true']['data'].to(device),
                'mask': batch['true']['mask'].to(device),
            },
            'reco': {
                'data': batch['reco']['data'].to(device),
                'mask': batch['reco']['mask'].to(device),
            },
            'params' : batch['params'].to(device),
        }
        if 'time' in batch.keys():
            batch_new['time'] = batch['time'].to(device)
        if 'mult' in batch.keys():
            batch_new['mult'] = batch['mult'].to(device)
        return batch_new

    def encode(self,batch):
        # Preprocessing #
        batch = self.apply_preprocessing(batch)
        # Embedding #
        x_enc,m_enc,x_dec,m_dec = self.apply_embeddings(batch)
        # Pass through transformer #
        x_enc = self.encoder(
            x = x_enc,
            mask = ~m_enc,
        )
        condition = self.decoder(
            x = x_dec,
            mask = ~m_dec,
            memory = x_enc,
            memory_mask = ~m_enc,
        )
        return condition

    def train_model(
        self,
        train_dataset: SurrogateDataset,
        valid_dataset: SurrogateDataset = None,
        batch_size: int = 64,
        n_epochs: int = 1,
        lr: float = 1e-3,
        teacher_forcing: float = 1.0,
        plotter: LossPlotting = None,
        loss_matching = None,
        num_workers = None,
    ):
        train_loader = DataLoader(
            dataset = train_dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers = num_workers,
            pin_memory = True,
        )
        if valid_dataset is None:
            valid_loader = None
        else:
            valid_loader = DataLoader(
                dataset = valid_dataset,
                batch_size = batch_size,
                shuffle = True,
                num_workers = num_workers,
                pin_memory = True,
            )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.to(self.device)

        print(f"Surrogate Training: {n_epochs=}, {lr=}, {batch_size=}, {teacher_forcing=}")
        for key,values in self.loss_factors.items():
            print (f'{key:15s} : {values.item():6.3f}')

        for epoch in range(n_epochs):
            # Training #
            self.train()
            train_losses = {
                'total' : torch.zeros(len(train_loader)),
                'flow'  : torch.zeros(len(train_loader)),
            }
            if self.multiplicity != 'none':
                train_losses['mult'] = torch.zeros(len(train_loader))
            if self.time_head is not None:
                train_losses['time'] = torch.zeros(len(train_loader))
            if loss_matching is not None:
                train_losses['matching'] = torch.zeros(len(train_loader))
            for batch_idx, batch in tqdm(enumerate(train_loader),total=len(train_loader),desc='Training batches',leave=False):
                # Move to device #
                batch = self.move_batch(batch,self.device)
                # Process #
                flow_losses,mult_losses,time_losses = self(batch)
                # Record #
                tot_losses = []
                train_losses['flow'][batch_idx] = flow_losses.mean()
                if 'flow' in self.loss_factors.keys():
                    factor = self.loss_factors['flow']
                else:
                    factor = 1.
                tot_losses.append(factor * flow_losses)
                if time_losses is not None:
                    train_losses['time'][batch_idx] = time_losses.mean()
                    if 'time' in self.loss_factors.keys():
                        factor = self.loss_factors['time']
                    else:
                        factor = 1.
                    tot_losses.append(factor * time_losses)
                if mult_losses is not None:
                    train_losses['mult'][batch_idx] = mult_losses.mean()
                    if 'mult' in self.loss_factors.keys():
                        factor = self.loss_factors['mult']
                    else:
                        factor = 1.
                    tot_losses.append(factor * mult_losses)
                if loss_matching is not None:
                    self.eval()
                    samples,samples_mask,_ = self.sample(batch)
                    self.train()
                    loss_match_values = loss_matching(
                        batch['reco']['data'],
                        batch['reco']['mask'],
                        samples,
                        samples_mask,
                    )['total']
                    train_losses['matching'][batch_idx] = loss_match_values.mean()
                    if 'matching' in self.loss_factors.keys():
                        factor = self.loss_factors['matching']
                    else:
                        factor = 1.
                    tot_losses.append(factor * loss_match_values)
                tot_loss = sum(tot_losses).mean()
                train_losses['total'][batch_idx] = tot_loss.item()
                # Optimizer step #
                self.optimizer.zero_grad()
                tot_loss.backward()
                self.optimizer.step()
            # Validation #
            self.eval()
            if valid_loader is not None:
                valid_losses = {
                    'total' : torch.zeros(len(valid_loader)),
                    'flow'  : torch.zeros(len(valid_loader)),
                }
                if self.time_head is not None:
                    valid_losses['time'] = torch.zeros(len(valid_loader))
                if self.multiplicity != 'none':
                    valid_losses['mult'] = torch.zeros(len(valid_loader))
                if loss_matching is not None:
                    valid_losses['matching'] = torch.zeros(len(valid_loader))
                for batch_idx, batch in tqdm(enumerate(valid_loader),total=len(valid_loader),desc='Validation batches',leave=False):
                    # Move to device #
                    batch = self.move_batch(batch,self.device)
                    # Process #
                    with torch.no_grad():
                        flow_losses,mult_losses,time_losses = self(batch)
                    # Record #
                    tot_losses = []
                    valid_losses['flow'][batch_idx] = flow_losses.mean()
                    if 'flow' in self.loss_factors.keys():
                        factor = self.loss_factors['flow']
                    else:
                        factor = 1.
                    tot_losses.append(factor * flow_losses)
                    if time_losses is not None:
                        valid_losses['time'][batch_idx] = time_losses.mean()
                        if 'time' in self.loss_factors.keys():
                            factor = self.loss_factors['time']
                        else:
                            factor = 1.
                        tot_losses.append(factor * time_losses)
                    if mult_losses is not None:
                        valid_losses['mult'][batch_idx] = mult_losses.mean()
                        if 'mult' in self.loss_factors.keys():
                            factor = self.loss_factors['mult']
                        else:
                            factor = 1.
                        tot_losses.append(factor * mult_losses)
                    if loss_matching is not None:
                        with torch.no_grad():
                            samples,samples_mask,_ = self.sample(batch)
                        loss_match_values = loss_matching(
                            batch['reco']['data'],
                            batch['reco']['mask'],
                            samples,
                            samples_mask,
                        )['total']
                        valid_losses['matching'][batch_idx] = loss_match_values.mean()
                        if 'matching' in self.loss_factors.keys():
                            factor = self.loss_factors['matching']
                        else:
                            factor = 1.
                        tot_losses.append(factor * loss_match_values)
                    tot_loss = sum(tot_losses).mean()
                    valid_losses['total'][batch_idx] = tot_loss.item()
            else:
                valid_losses = {
                    'total' : torch.zeros(1),
                    'flow' : torch.zeros(1),
                }
                if self.time_head is not None:
                    valid_losses['time'] = torch.zeros(1)
                if self.multiplicity != 'none':
                    valid_losses['mult'] = torch.zeros(1)

            # Printout #
            s = f"Surrogate epoch = {epoch:3d} - Loss = {train_losses['total'].mean():+10.5f} [{valid_losses['total'].mean():+10.5f}] : "
            for name in train_losses.keys():
                if name == 'total':
                    continue
                if name in self.loss_factors.keys():
                    factor = self.loss_factors[name]
                else:
                    factor = 1.
                s += f"{name} = {factor.item(): .3f} x {train_losses[name].mean():+10.5f} [{valid_losses[name].mean():+10.5f}] + "
            print (s)
            self.surrogate_loss.append(valid_losses['total'].mean())

            # Plotter #
            if plotter is not None:
                plotter.add_lr_value(lr)
                for name,losses in train_losses.items():
                    plotter.add_train_value(name,losses.mean().item())
                for name,losses in valid_losses.items():
                    plotter.add_valid_value(name,losses.mean().item())

    def sample(self,batch,thresholds):
        # Preprocessing #
        batch = self.apply_preprocessing(batch)
        # Sampling #
        samples, samples_mask, times = self._sample(batch,thresholds)
        # Undo preprocessing #
        samples = samples * self.stds['reco'] + self.means['reco']
        return samples,samples_mask,times

    @torch.no_grad()
    def inference(
        self,
        dataset,
        batch_size,
        thresholds,
        oversampling: int = 1,
    ):
        self.eval()
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        if self.multiplicity == 'none':
            max_seq_len = dataset[0]['reco']['data'].shape[0]
        else:
            max_seq_len = self.max_seq_len
        n_features  = sum(
            [
                len(idxs)
                for idxs in self.feature_dict.values()
            ]
        )
        all_samples = torch.zeros((oversampling, len(dataset), max_seq_len, n_features))
        all_masks = torch.zeros((oversampling, len(dataset), max_seq_len)) > 0
        all_times = torch.zeros((oversampling, len(dataset), 1))

        for io in tqdm(range(oversampling),desc='Oversample',total=oversampling,leave=True,position=0):
            for batch_idx, batch in tqdm(enumerate(data_loader),desc='Sample',total=len(data_loader),leave=False,position=1):
                # Move to device #
                batch = self.move_batch(batch,self.device)
                # Sample #
                samples,mask,times = self.sample(batch,thresholds)
                # Record #
                ia = batch_idx * batch_size
                ib = (batch_idx + 1) * batch_size
                all_samples[io,ia:ib] = samples.cpu()
                all_masks[io,ia:ib] = mask.cpu()
                if times is not None:
                    all_times[io,ia:ib] = times.cpu()

        return all_samples,all_masks,all_times

    def compute_log_prob_thresholds(self,dataset,batch_size,quantile):
        self.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_log_probs = []
        for batch in tqdm(loader,total=len(loader),desc='Batches'):
            batch = self.move_batch(batch, self.device)
            all_log_probs.append(self.log_prob(batch))
        return torch.quantile(torch.cat(all_log_probs,dim=0), quantile).item()



class SurrogateFusion(Surrogate):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.multiplicity = 'none'

        # Transformer #
        self.encoder = Encoder(
            dim_embed = self.dim_embed,
            nhead = 4,
            num_layers = 4,
            expansion_factor = 4,
            dropout = 0.,
        )
        self.decoder = Decoder(
            dim_embed = self.dim_embed,
            nhead = 4,
            num_layers = 6,
            expansion_factor = 4,
            dropout = 0.,
            is_causal = False,
        )

        self.time_embed = TimeEmbedding(self.dim_embed)
        self.projection = nn.Sequential(
            nn.Linear(self.dim_embed * 2,64),
            nn.GELU(),
            nn.Linear(64,32),
            nn.GELU(),
            nn.Linear(32,len(self.order)),
        )
        self.time_head = None

        self.optimizer = torch.optim.RAdam(self.parameters(), lr=0.01)
#        # Multiplicity #
#        if self.multiplicity == 'hard':
#            self.mult_hard_encoding = CrossEncoder(
#                dim_embed = self.dim_embed,
#                nhead = 4,
#                num_layers = 3,
#                expansion_factor = 2,
#                dropout = 0.1,
#            )
#            self.mult_hard_head = ClassifierMultiplicity(
#                dim_embed = self.dim_embed,
#                max_seq_len = self.max_seq_len,
#                dropout = 0.1,
#            )
#        if self.multiplicity == 'soft':
#            self.mult_soft_head = ClassifierExist(
#                dim_embed = self.dim_embed,
#                alpha = 0.25,
#                gamma = 2,
#                max_seq_len = self.max_seq_len,
#                dropout = 0.1,
#            )
#
#        # Timing #
#        if 'time' in shapes.keys():
#            self.time_encoding = CrossEncoder(
#                dim_embed = self.dim_embed,
#                nhead = 4,
#                num_layers = 3,
#                expansion_factor = 2,
#                dropout = 0.2,
#            )
#            self.time_head = zuko.flows.NSF(
#                features = 1,
#                context = self.dim_embed,
#                bins = 16,
#                transforms = 3,
#                randperm = True,
#                hidden_features = [128,128],
#            )
#        else:
#            self.time_encoding = None
#            self.time_head = None

    def apply_embeddings(self,batch):
        x_enc = torch.cat(
            [
                self.embeddings['params'](
                    batch['params']
                ),
                self.embeddings['true'](
                    batch['true']['data']
                ),
            ],
            dim = 1,
        )
        m_enc = torch.cat(
            [
                torch.full(
                    (batch['true']['mask'].shape[0],1),
                    fill_value = True,
                ).to(batch['true']['mask'].device),
                batch['true']['mask'],
            ],
            dim = 1,
        )
        x_dec = self.embeddings['reco'](batch['reco']['data'])
        m_dec = batch['reco']['mask']
        return x_enc,m_enc,x_dec,m_dec


    def forward(self,batch):
        # Preprocessing #
        batch = self.apply_preprocessing(batch)

        # Prepare for CFM #
        x1 = batch['reco']['data'].clone()
        m = batch['reco']['mask'].clone()
        x0 = torch.randn_like(x1,device=x1.device)
        t = torch.rand(x1.shape[0], 1, 1, device=x1.device).repeat_interleave(x1.shape[1],dim=1)
        x = (1 - t) * x0 + t * x1
        batch['reco']['data'] = x

        # Embedding #
        x_enc,m_enc,x_dec,m_dec = self.apply_embeddings(batch)

        # Pass through transformer #
        x_enc = self.encoder(
            x = x_enc,
            mask = ~m_enc,
        )
        condition = self.decoder(
            x = x_dec,
            mask = ~m_dec,
            memory = x_enc,
            memory_mask = ~m_enc,
        )

        # Projection #
        vy = self.projection(
            torch.cat(
                [
                    condition,
                    self.time_embed(t),
                ],
                dim = -1,
            )
        )

        # Loss #
        vt = x1 - x0
        v_losses = (((vt - vy) ** 2).mean(dim=-1) * m).sum(dim=-1)
        return v_losses,None,None


    def _sample(self,batch,replace_mask=None):
        """ Sample without preprocessing """
        # Embedding #
        x_enc, m_enc, _, m_dec = self.apply_embeddings(batch)

        # Pass through encoder #
        x_enc = self.encoder(
            x = x_enc,
            mask = ~m_enc,
        )

        # Solve ODE #
        def ode_func(t, x):
            t = torch.full(
                (x.shape[0], 1, 1),
                t.item(),
                device=x.device
            ).repeat_interleave(x.shape[1],dim=1)
            x = self.embeddings['reco'](x)
            c = self.decoder(
                x = x,
                mask = ~m_dec,
                memory = x_enc,
                memory_mask = ~m_enc,
            )
            v = self.projection(
                torch.cat(
                    [
                        c,
                        self.time_embed(t),
                    ],
                    dim = -1,
                )
            )
            return v

        x0 = torch.randn_like(batch['reco']['data'],device=batch['reco']['data'].device)
        t_span = torch.tensor([0.0, 1.0], device=x0.device)
        traj = odeint(
            ode_func,
            x0,
            t_span,
            method="dopri5",
            rtol=1e-6,
            atol=1e-6,
            #method="rk4",
            #options={"step_size": 0.01},
        )

        samples = traj[-1]
        samples_mask = m_dec

        return samples,samples_mask,None
#        # Timing #
#        if self.time_head is not None:
#            time_enc = self.time_encoding(
#                slots = self.time_token.expand(x_enc.shape[0], -1, -1),
#                encoder_out = x_enc,
#                encoder_padding_mask = ~m_enc,
#            ).squeeze(dim=1)
#            times = self.time_head(time_enc).rsample()
#            times = times * self.stds['time'] + self.means['time']
#        else:
#            times = None
#        # Make empty decoder input (with start token) #
#        if self.multiplicity =='hard' and replace_mask is None:
#            # Hard classifier : we sample the sampled mask, and replace decoder mask
#            mult_enc = self.mult_hard_encoding(
#                slots = self.mult_hard_token.expand(x_enc.shape[0], -1, -1),
#                encoder_out = x_enc,
#                encoder_padding_mask = ~m_enc,
#            ).squeeze(dim=1)
#            samples_mask = self.mult_hard_head.sample(mult_enc)
#            m_dec = torch.cat(
#                [
#                    torch.full( # add start token mask
#                        (samples_mask.shape[0],1),
#                        fill_value = True,
#                    ).to(mask_target.device),
#                    samples_mask,
#                ],
#                dim = 1,
#            )
#            N = self.max_seq_len
#        elif self.multiplicity == 'soft' and replace_mask is None:
#            # Soft classifier : we use unmasked particles and predict later
#            m_dec = torch.full(
#                (
#                    m_dec.shape[0],
#                    self.max_seq_len,
#                ),
#                fill_value = True,
#            ).to(m_dec.device)
#            N = self.max_seq_len
#        else:
#            samples_mask = mask_target
#            N = mask_target.shape[1]
#        samples = []
#        x_dec = [x_dec[:,0:1,:]] # only keep start token
#        # Sequential generation #
#        for i in range(N):
#            # Pass through decoder, obtain new context #
#            condition = self.decoder(
#                x = torch.cat(x_dec,dim=1),
#                mask = ~m_dec[:,:len(x_dec)],
#                memory = x_enc,
#                memory_mask = ~m_enc,
#            )[:,i,:]
#            # Create placeholder particle
#            particle = torch.zeros(
#                (
#                    batch['reco']['data'].shape[0],
#                    batch['reco']['data'].shape[2],
#                )
#            ).to(self.device)
#            # Flow #
#            particle = self.flow.sample(condition)[:,self.order.to(particle.device)]
#            samples.append(particle)
#            x_dec.append(self.embeddings['reco'](particle).unsqueeze(1))
#        # Soft classifier for existence sampling #
#        if self.multiplicity == 'soft' and replace_mask is None:
#            condition = self.decoder(
#                x = torch.cat(x_dec[:-1],dim=1),
#                mask = ~m_dec,
#                memory = x_enc,
#                memory_mask = ~m_enc,
#            )
#            samples_mask = self.mult_soft_head.sample(condition)
#        # Stack and zero-out missing particles #
#        samples = torch.stack(samples,dim=1)
#        samples = (samples * samples_mask.unsqueeze(-1))
#        # Sort so that True particles (present) are before False (absent)
#        idx = torch.argsort(samples_mask.int(),dim=1,descending=True,stable=True)
#        samples = torch.gather(
#            input = samples,
#            dim = 1,
#            index = idx.unsqueeze(-1).expand(-1, -1, samples.size(-1)),
#        )
#        samples_mask = samples_mask[
#            torch.arange(samples_mask.size(0), device=samples_mask.device)[:, None],
#            idx,
#        ]
#
#        return samples,samples_mask,times



class AutoRegressiveSurrogate(Surrogate):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Transformer #
        self.encoder = Encoder(
            dim_embed = self.dim_embed,
            nhead = 4,
            num_layers = 4,
            expansion_factor = 2,
            dropout = 0.,
        )
        self.decoder = Decoder(
            dim_embed = self.dim_embed,
            nhead = 4,
            num_layers = 4,
            expansion_factor = 2,
            dropout = 0.,
            is_causal = True,
        )
        self.start_token = nn.Parameter(
            torch.zeros(1,1,self.dim_embed),
            requires_grad = True,
        )


    def apply_embeddings(self,batch):
        x_enc = torch.cat(
            [
                self.embeddings['params'](
                    batch['params']
                ),
                self.embeddings['true'](
                    batch['true']['data']
                ),
            ],
            dim = 1,
        )
        m_enc = torch.cat(
            [
                torch.full(
                    (batch['true']['mask'].shape[0],1),
                    fill_value = True,
                ).to(batch['true']['mask'].device),
                batch['true']['mask'],
            ],
            dim = 1,
        )
        x_dec = torch.cat(
            [
                self.start_token.expand(batch['reco']['data'].shape[0], -1, -1),
                self.embeddings['reco'](
                    batch['reco']['data'][:,:-1]
                ),
            ],
            dim = 1,
        )
        m_dec = torch.cat(
            [
                torch.full(
                    (batch['reco']['mask'].shape[0],1),
                    fill_value = True,
                ).to(batch['reco']['mask'].device),
                batch['reco']['mask'][:,:-1],
            ],
            dim = 1,
        )
        return x_enc,m_enc,x_dec,m_dec

    def forward(self,batch):
        # Preprocessing #
        batch = self.apply_preprocessing(batch)
        # Select (preprocessed) targets #
        reco_target = batch['reco']['data'].clone()
        mask_target = batch['reco']['mask'].clone()
        # Embedding #
        x_enc,m_enc,x_dec,m_dec = self.apply_embeddings(batch)
        # Pass through transformer #
        x_enc = self.encoder(
            x = x_enc,
            mask = ~m_enc,
        )
        condition = self.decoder(
            x = x_dec,
            mask = ~m_dec,
            memory = x_enc,
            memory_mask = ~m_enc,
        )
        # Multiplicity classifiers #
        if self.multiplicity == 'hard':
            mult_losses = self.mult_hard.loss(batch)
        elif self.multiplicity == 'soft':
            y = self.mult_soft_head(condition)
            mult_losses = self.mult_soft_head.loss(y.squeeze(-1),mask_target)
        else:
            mult_losses = None

        # Process time flow #
        if 'time' in batch.keys():
            time_enc = self.time_encoding(
                x = x_enc,
                mask = ~m_enc,
            ).squeeze(dim=1)
            time_losses = - self.time_head(time_enc).log_prob(batch['time'])
        else:
            time_losses = None

        # Flow
        flow_losses = self.flow.loss(
            c = condition,
            y = torch.index_select(
                input = reco_target,
                dim = -1,
                index = self.order.to(reco_target.device),
            ),
            m = mask_target,
        )

        return flow_losses, mult_losses, time_losses

    def _sample_flow(self,c,thresh):
        y = torch.zeros(
            (
                c.shape[0],
                len(self.order),
            ),
            device = c.device,
        )
        log_probs = torch.full(
            (
                c.shape[0],
            ),
            fill_value = -torch.inf,
            device = c.device,
        )
        while thresh is None or (log_probs < thresh).sum() > 0:
            if thresh is not None:
                mask = log_probs < thresh
            else:
                mask = torch.ones_like(log_probs) > 0
            print ('before self.flow.sample')
            y[mask] = self.flow.sample(c[mask])
            if thresh is None:
                break
            log_probs[mask] = self.flow.log_prob(c[mask],y[mask])
        return y


    def _sample(self,batch,thresholds=None):
        """ Sample without preprocessing """
        # Get targets #
        reco_target = batch['reco']['data'].clone()
        mask_target = batch['reco']['mask'].clone()
        # Multiplicity #
        if self.multiplicity =='hard':
            # Hard classifier : we sample the sampled mask, and replace decoder mask
            samples_mask = self.mult_hard.sample(batch)
            batch['mult'] = F.one_hot(samples_mask.sum(dim=-1)-1,self.max_seq_len)
            N = self.max_seq_len
        elif self.multiplicity == 'soft':
            # Soft classifier : we use unmasked particles and predict later
            m_dec = torch.full(
                (
                    m_dec.shape[0],
                    self.max_seq_len,
                ),
                fill_value = True,
            ).to(m_dec.device)
            N = self.max_seq_len
        else:
            samples_mask = mask_target
            N = mask_target.shape[1]

        # Embedding #
        x_enc, m_enc, x_dec, m_dec = self.apply_embeddings(batch)
        m_dec = torch.cat(
            [
                torch.full(
                    (
                        samples_mask.shape[0],
                        1,
                    ),
                    fill_value = True,
                ).to(mask_target.device),
                samples_mask,
            ],
            dim = 1,
        )
        # Pass through encoder #
        x_enc = self.encoder(
            x = x_enc,
            mask = ~m_enc,
        )
        # Timing #
        if self.time_head is not None:
            time_enc = self.time_encoding(
                slots = self.time_token.expand(x_enc.shape[0], -1, -1),
                encoder_out = x_enc,
                encoder_padding_mask = ~m_enc,
            ).squeeze(dim=1)
            times = self.time_head(time_enc).rsample()
            times = times * self.stds['time'] + self.means['time']
        else:
            times = None
        # Make empty decoder input (with start token) #
        samples = []
        x_dec = [x_dec[:,0:1,:]] # only keep start token
        # Sequential generation #
        for i in range(N):
            # Pass through decoder, obtain new context #
            condition = self.decoder(
                x = torch.cat(x_dec,dim=1),
                mask = ~m_dec[:,:len(x_dec)],
                memory = x_enc,
                memory_mask = ~m_enc,
            )[:,i,:]
            # Flow #
            if thresholds is None:
                threshold = None
            else:
                threshold = thresholds[min(i,len(thresholds)-1)]
            particle = self._sample_flow(
                c = condition,
                thresh = threshold,
            )[:,self.order.to(condition.device)]
            samples.append(particle)
            x_dec.append(self.embeddings['reco'](particle).unsqueeze(1))
        # Soft classifier for existence sampling #
        if self.multiplicity == 'soft' and replace_mask is None:
            condition = self.decoder(
                x = torch.cat(x_dec[:-1],dim=1),
                mask = ~m_dec,
                memory = x_enc,
                memory_mask = ~m_enc,
            )
            samples_mask = self.mult_soft_head.sample(condition)
        # Stack and zero-out missing particles #
        samples = torch.stack(samples,dim=1)
        samples = (samples * samples_mask.unsqueeze(-1))
        # Sort so that True particles (present) are before False (absent)
        idx = torch.argsort(samples_mask.int(),dim=1,descending=True,stable=True)
        samples = torch.gather(
            input = samples,
            dim = 1,
            index = idx.unsqueeze(-1).expand(-1, -1, samples.size(-1)),
        )
        samples_mask = samples_mask[
            torch.arange(samples_mask.size(0), device=samples_mask.device)[:, None],
            idx,
        ]

        return samples,samples_mask,times

    @torch.no_grad()
    def log_prob(self,batch):
        # Preprocessing #
        batch = self.apply_preprocessing(batch)
        # Select (preprocessed) targets #
        reco_target = batch['reco']['data'].clone()
        mask_target = batch['reco']['mask'].clone()
        # Embedding #
        x_enc,m_enc,x_dec,m_dec = self.apply_embeddings(batch)
        # Pass through transformer #
        x_enc = self.encoder(
            x = x_enc,
            mask = ~m_enc,
        )
        condition = self.decoder(
            x = x_dec,
            mask = ~m_dec,
            memory = x_enc,
            memory_mask = ~m_enc,
        )
        log_probs = self.flow.log_prob(
            c = condition,
            y = torch.index_select(
                input = reco_target,
                dim = -1,
                index = self.order.to(reco_target.device),
            ),
            m = mask_target,
        )
        return log_probs

    def compute_log_prob_thresholds(self,dataset,batch_size,quantile,single=True):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        log_probs = []
        masks = []
        for batch in tqdm(loader,total=len(loader),desc='Batches'):
            masks.append(batch['reco']['mask'])
            batch = self.move_batch(batch,self.device)
            log_probs.append(self.log_prob(batch).cpu())
        log_probs = torch.cat(log_probs,dim=0)
        masks = torch.cat(masks,dim=0)
        thresholds = []
        colors = plt.cm.rainbow(np.linspace(0, 1, log_probs.shape[1]))
        bins = np.linspace(
            log_probs[masks].min(),
            log_probs[masks].max(),
            40,
        )
        if single:
            thresholds = torch.tensor([torch.quantile(log_probs[masks], quantile).item()])
        else:
            thresholds = []
            for i in range(log_probs.shape[1]):
                if masks[:,i].sum() > 100:
                    thresholds.append(torch.quantile(log_probs[:,i][masks[:,i]], quantile).item())
            thresholds = torch.tensor(thresholds)
        return thresholds



class INNHead(nn.Module):
    def __init__(self, dim_in, dim_cond):
        super().__init__()

        self.dim_in = dim_in
        self.dim_cond = dim_cond

        self.flow = zuko.flows.NSF(
            features = self.dim_in,
            context = self.dim_cond,
            bins = 8,
            transforms = 8,
            randperm = True,
            passes = None,
            hidden_features = [128,128],
        )

        self.optimizer = torch.optim.RAdam(self.parameters(), lr=0.01)

    def forward(self,x,c):
        return self.flow(c).log_prob(x)

    def loss(self,c,y,m):
        flow_losses = - (self(y,c) * m).sum(dim=-1)
        return flow_losses

    def log_prob(self,c,y,m=None):
        if m is None:
            m = torch.ones(y.shape[:-1],device=y.device)
        return self(y,c) * m

    def sample(self,c,N=1):
        samples = self.flow(c).rsample((N,))
        if N == 1:
            samples = samples[0]
        return self.flow(c).rsample((N,))

    def inverse(self,c,x):
        return self.flow(c).transform(x)

class MixtureINNHead(nn.Module):
    def __init__(self, dim_in, dim_cond, components):
        super().__init__()

        self.dim_in = dim_in
        self.dim_cond = dim_cond
        self.components = components

        self.flows = nn.ModuleList(
            [
                zuko.flows.NSF(
                    features = self.dim_in,
                    context = self.dim_cond,
                    bins = 8,
                    transforms = 3,
                    randperm = True,
                    passes = None,
                    hidden_features = [128,128],
                )
                for _ in range(self.components)
            ]
        )
        self.gate = nn.Sequential(
            nn.Linear(self.dim_cond, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, self.components)
        )

        self.optimizer = torch.optim.RAdam(self.parameters(), lr=0.01)

    def forward(self,x,c):
        weights = torch.softmax(self.gate(c), dim=-1)

        log_probs = []
        for flow in self.flows:
            log_probs.append(flow(c).log_prob(x))

        log_probs = torch.stack(log_probs, dim=-1)

        return weights, log_probs

    def loss(self,y,c,m):
        weights, log_probs = self(y,c)
        log_weights = torch.log(weights + 1e-8)

        # log sum exp for numerical stability
        nll = - torch.logsumexp(log_probs + log_weights, dim=-1)
        # entropy to avoid collapse
        entropy = - (weights * log_weights).sum(-1).mean()

        loss = (nll*m).sum(dim=-1) - 0.01 * entropy

        return loss

    def sample(self,c,N=1):
        logits = self.gate(c)

        weights = torch.nn.functional.gumbel_softmax(
            logits, tau=0.5, hard=True,
        )

        x = 0.
        for k, flow in enumerate(self.flows):
            xk = flow(c).rsample((N,))
            x = x + weights[...,k:k+1] * xk
        if N == 1:
            x = x[0]
        return x

    def log_prob(self,c,y,m=None):
        pass

    def inverse(self,c,x):
        pass

class AcceptanceINNHead(nn.Module):
    def __init__(self, dim_in, dim_cond):
        super().__init__()

        self.dim_in = dim_in
        self.dim_cond = dim_cond

        self.flow = zuko.flows.NSF(
            features = self.dim_in,
            context = self.dim_cond,
            bins = 8,
            transforms = 8,
            randperm = True,
            passes = None,
            hidden_features = [128,128],
        )
        self.accept = nn.Sequential(
            nn.Linear(self.dim_in + self.dim_cond, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

        self.optimizer = torch.optim.RAdam(self.parameters(), lr=0.01)

    def forward(self,x,c):
        dist = self.flow(c)
        log_prob = dist.log_prob(x)
        z = dist.transform(x)
        log_a = torch.nn.functional.logsigmoid(
            self.accept(torch.cat((c,z),dim=-1)).squeeze(-1)
        )

        # normalisation constant Z via Monte Carlo
        #with torch.no_grad():
        #    B,S,F = z.shape
        #    n_mc = 2000
        #    z_mc = torch.randn(n_mc, B, S, F, device=z.device)
        #    c_mc = c.unsqueeze(0).expand(n_mc, -1, -1, -1)

        #    # flatten n_mc and B and S together for the net
        #    z_mc_flat = z_mc.reshape(n_mc * B * S, F)
        #    c_mc_flat = c_mc.reshape(n_mc * B * S, -1) if c.dim() == 3 \
        #                else c_mc.reshape(n_mc * B * S, c.shape[-1])

        #    a_mc = torch.sigmoid(
        #        self.accept(torch.cat((c_mc,z_mc),dim=-1))
        #    ).reshape(n_mc, B, S)

        #    log_Z = torch.log(a_mc.mean(dim=0) + 1e-8)

        return log_prob + log_a #- log_Z

    def loss(self,y,c,m):
        log_prob = self(y,c)
        return - (log_prob * m).sum(dim=-1)

    def sample(self,c,N=1):
        print ('sample')
        dist = self.flow(c)

        samples = torch.empty(flat_shape, device=device)

        z = torch.zeros((*c.shape[:-1],self.dim_in),device=c.device)
        remaining = torch.zeros(*c.shape[:-1],dtype=torch.bool,device=c.device)
        from IPython import embed; embed()

#        while remaining.any():
#
#
#
#        z = dist.base.sample((N,))
#        if N == 1:
#            z = z.squeeze(0)
#        a = torch.sigmoid(
#            self.accept_net(
#                torch.cat(
#                    (
#                        z,
#                        c.expand(N, *c.shape[1:]),
#                    ),
#                    dim = -1,
#                ).squeeze(-1)
#            )
#        )
#        accepted = torch.rand(n, device=c.device) < a
#
#        while not accepted.all():
#            n_rejected = (~accepted).sum().item()
#            #z_new = torch.randn(n_rejected, self.dim_z, device=c.device)
#            z_new = dist.base.sample((N,))
#            if N == 1:
#                z_new = z_new.squeeze(0)
#            a_new = torch.sigmoid(
#                self.accept_net(
#                    torch.cat(
#                        (
#                            z_new,
#                            c.expand(N, *c.shape[1:]),
#                        ),
#                        dim = -1,
#                    ).squeeze(-1)
#                )
#            )
#            #a_new = torch.sigmoid(self.accept_net(z_new, c_exp).squeeze(-1))
#            newly_accepted = torch.rand(n_rejected, device=c.device) < a_new
#
#            # write accepted ones back into the rejected slots
#            rejected_idx = (~accepted).nonzero(as_tuple=True)[0]
#            accepted_idx = rejected_idx[newly_accepted]
#            z[accepted_idx] = z_new[newly_accepted]
#            accepted[accepted_idx] = True
#

    def log_prob(self,c,y,m=None):
        pass

    def inverse(self,c,x):
        pass


#    class FlowWithLARS(nn.Module):
#    def __init__(self, flow, dim):
#        super().__init__()
#        self.flow = flow                         # your existing zuko flow, frozen or not
#        self.accept_net = nn.Sequential(
#            nn.Linear(dim, 128),
#            nn.SiLU(),
#            nn.Linear(128, 128),
#            nn.SiLU(),
#            nn.Linear(128, 1)
#        )
#
#    def log_prob(self, x, c):
#        # standard zuko log prob — unchanged
#        dist = self.flow(c)
#        log_p_flow = dist.log_prob(x)            # (B,)
#
#        # encode x to z to evaluate acceptance
#        z = dist.transform(x)                    # (B, dim)
#        log_a = torch.nn.functional.logsigmoid(
#            self.accept_net(z).squeeze(-1)
#        )                                        # (B,)
#
#        # normalisation constant Z via Monte Carlo
#        with torch.no_grad():
#            z_mc = torch.randn(2000, z.shape[-1], device=x.device)
#            log_Z = torch.log(
#                torch.sigmoid(self.accept_net(z_mc)).mean() + 1e-8
#            )
#
#        return log_p_flow + log_a - log_Z        # (B,)
#
#    @torch.no_grad()
#    def sample(self, c, n):
#        base = torch.distributions.Independent(
#            torch.distributions.Normal(
#                torch.zeros(z_dim, device=c.device),
#                torch.ones(z_dim, device=c.device)
#            ), 1
#        )
#        # rejection sampling in latent space
#        samples = []
#        while len(samples) < n:
#            z = torch.randn(n, self.dim, device=c.device)
#            a = torch.sigmoid(self.accept_net(z).squeeze(-1))
#            u = torch.rand_like(a)
#            accepted = z[u < a]
#            if len(accepted) > 0:
#                samples.append(accepted)
#
#        z_accepted = torch.cat(samples)[:n]
#        return self.flow(c).transform.inv(z_accepted)   # decode


#class LARSBase(nn.Module):
# https://arxiv.org/pdf/2110.15828
#    """Learned Accept/Reject Sampling base distribution."""
#    def __init__(self, dim, T=10):
#        super().__init__()
#        self.T = T  # max rejections before forced accept
#        # acceptance network: R^dim → [0,1]
#        self.accept_net = nn.Sequential(
#            nn.Linear(dim, 128),
#            nn.SiLU(),
#            nn.Linear(128, 128),
#            nn.SiLU(),
#            nn.Linear(128, 1),
#            nn.Sigmoid()
#        )
#
#    def log_prob(self, z):
#        """Log prob of resampled base at z."""
#        log_gaussian = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(-1)
#        log_accept = torch.log(self.accept_net(z).squeeze(-1) + 1e-8)
#        # normalisation constant Z estimated via Monte Carlo
#        z_mc = torch.randn(1000, z.shape[-1], device=z.device)
#        log_Z = torch.log(self.accept_net(z_mc).mean() + 1e-8)
#        return log_gaussian + log_accept - log_Z
#
#    def sample(self, n, device='cpu'):
#        samples = []
#        while len(samples) < n:
#            for t in range(self.T):
#                z = torch.randn(n, self.dim, device=device)
#                a = self.accept_net(z).squeeze(-1)
#                u = torch.rand_like(a)
#                accepted = z[u < a]
#                samples.append(accepted)
#                if len(samples) >= n:
#                    break
#        return torch.cat(samples)[:n]
# def training_loss(flow, lars_base, x, c):
#     # encode x to latent
#     z, log_det = flow.encode(x, condition=c)
#     # log prob under LARS base
#     log_p_z = lars_base.log_prob(z)
#     # full log prob
#     log_p_x = log_p_z + log_det
#     return -log_p_x.mean()
# post-hoc: train only the acceptance network
# freeze the flow, learn a_φ to reject bridge regions
#
#def train_acceptance_net(flow, accept_net, X_train, C_train, n_epochs=100):
#    opt = torch.optim.Adam(accept_net.parameters(), lr=1e-3)
#    flow.eval()
#
#    for epoch in range(n_epochs):
#        for x, c in dataloader(X_train, C_train):
#            with torch.no_grad():
#                z, _ = flow.encode(x, condition=c)
#
#            # maximise acceptance of true data latents
#            a = accept_net(z).squeeze(-1)
#
#            # also sample from N(0,1) and minimise their acceptance
#            z_random = torch.randn_like(z)
#            a_random = accept_net(z_random).squeeze(-1)
#
#            loss = -torch.log(a + 1e-8).mean() + torch.log(a_random + 1e-8).mean()
#            opt.zero_grad()
#            loss.backward()
#            opt.step()

class SurrogateMixtureINN(AutoRegressiveSurrogate):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.flow = MixtureINNHead(
            dim_in = len(self.order),
            dim_cond = self.dim_embed,
            #components = self.max_seq_len,
            components = 10,
        )

        self.optimizer = torch.optim.RAdam(self.parameters(), lr=0.01)

class SurrogateAcceptanceINN(AutoRegressiveSurrogate):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.flow = AcceptanceINNHead(
            dim_in = len(self.order),
            dim_cond = self.dim_embed,
        )

        self.optimizer = torch.optim.RAdam(self.parameters(), lr=0.01)



class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim)
        )

    def forward(self, t):
        """
        t: (batch, 1)
        returns: (batch, dim)
        """
        half_dim = self.dim // 2

        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=t.device) / (half_dim - 1)
        )

        args = t * freqs  # (batch, half_dim)

        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class CFMHead(nn.Module):
    def __init__(self, dim_in, dim_cond, dim_time, dim_hidden):
        super().__init__()

        self.dim_in = dim_in
        self.dim_cond = dim_cond
        self.dim_time = dim_time

        self.time_embed = TimeEmbedding(self.dim_time)

        self.embedding = nn.Linear(self.dim_in,dim_hidden)

        self.net = nn.Sequential(
            nn.Linear(dim_hidden + self.dim_time + self.dim_cond, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, self.dim_in)
        )

        #self._sample_args = {
        #    'method' : "dopri5",
        #    'rtol': 1e-6,
        #    'atol': 1e-6,
        #}
        self._sample_args = {
            'method' : "rk4",
            'options' : {"step_size": 0.01},
        }


    def forward(self, x, c, t):
        inp = torch.cat(
            [
                self.embedding(x),
                c,
                self.time_embed(t),

            ],
            dim = -1,
        )
        return self.net(inp)

    def loss(self,c,y,m):
        # Modify target to do the velocity target #
        x1 = y
        x0 = torch.randn_like(x1,device=x1.device)
        t = torch.rand(x1.shape[0], 1, 1, device=x1.device).repeat_interleave(x1.shape[1],dim=1)
        x = (1 - t) * x0 + t * x1
        vt = x1 - x0
        vy = self(x,c,t)
        v_losses = (((vt - vy) ** 2).mean(dim=-1) * m).sum(dim=-1)
        return v_losses

    def log_prob(self,c,y,m=None):
        if m is None:
            m = torch.ones(y.shape[:-1],device=y.device)

        params = dict(self.named_parameters())
        buffers = dict(self.named_buffers())

        def net_fn(x, c, t_full):
            return functional_call(self, (params, buffers), (x, c, t_full))

        def ode_func(t, state):
            x, log_det = state
            t_full = torch.full((*x.shape[:-1], 1), t.item(), device=x.device)

            noise = torch.randn_like(x)

            # vjp_fn computes vector-Jacobian products without a full backward pass
            v, vjp_fn = torch.func.vjp(lambda x_: net_fn(x_, c, t_full), x)
            div_estimate = (vjp_fn(noise)[0] * noise).sum(dim=-1, keepdim=True)

            return v.detach(), -div_estimate

#        def ode_func(t, state):
#            x, log_det = state
#            t_full = torch.full((*x.shape[:-1], 1), t.item(), device=x.device)
#
#            with torch.enable_grad():
#                x_in = x.detach().requires_grad_(True)
#                v = self(x_in, c, t_full)
#
#            noise = torch.randn_like(x_in)
#            vjp = torch.autograd.grad(v, x_in, grad_outputs=noise, create_graph=False)[0]
#            div_estimate = (vjp * noise).sum(dim=-1, keepdim=True)  # (B, 1)
#
#            return v.detach(), -div_estimate

        state0 = (y, torch.zeros(*y.shape[:-1], device=y.device))
        t_span = torch.tensor([1.0, 0.0], device=y.device)

        traj = odeint(
            ode_func,
            state0,
            t_span,
            method='euler',
            options={'step_size': 0.01},
            #**self._sample_args,
        )

        x0 = traj[0][-1]
        delta_log_det = traj[1][-1]

        log_p0 = torch.distributions.Normal(0, 1).log_prob(x0).sum(dim=-1)
        log_prob = (log_p0 + delta_log_det) * m

        return log_prob

    def _sample_base(self,shapes,threshold,device):
        if len(shapes) == 2:
            B, F = shapes
            flat_shape = (B, F)
        elif len(shapes) == 3:
            B, S, F = shapes
            flat_shape = (B * S, F)
        else:
            raise ValueError(f"Expected 2D or 3D shape, got {len(shape)}D")

        samples = torch.empty(flat_shape, device=device)
        remaining = torch.ones(flat_shape[0], dtype=torch.bool, device=device)

        while remaining.any():
            n = remaining.sum().item()
            candidates = torch.randn((n, F), device=device)
            valid = candidates.norm(dim=-1) < threshold
            idx = remaining.nonzero(as_tuple=True)[0]
            accepted = idx[valid]
            samples[accepted] = candidates[valid]
            remaining[accepted] = False

        return samples.reshape(shapes)

    def sample(self,c,N=1):
        def ode_func(t, x):
            t = torch.full(
                (*x.shape[:-1], 1),
                t.item(),
                device=x.device
            )
            v = self(x, c, t)
            return v

        #x0 = torch.randn((c.shape[0],self.dim_in),device=c.device)
        if N > 1:
            c = c.unsqueeze(dim=0).repeat_interleave(N,dim=0)
        x0 = self._sample_base((*c.shape[:-1],self.dim_in),threshold=torch.inf,device=c.device)
        t_span = torch.tensor([0.0, 1.0], device=c.device)
        traj = odeint(
            ode_func,
            x0,
            t_span,
            **self._sample_args,
        )
        x1 = traj[-1]
        return x1


    def inverse(self,c,x):
        def ode_func(t, x):
            t = torch.full(
                (*x.shape[:-1], 1),
                t.item(),
                device=x.device
            )
            v = self(x, c, t)
            return v

        t_span = torch.tensor([1.0, 0.0], device=x.device)
        traj = odeint(
            ode_func,
            x,
            t_span,
            **self._sample_args,
        )
        x0 = traj[-1]  # last step is t=0
        return x0



class SurrogateCFM(AutoRegressiveSurrogate):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.flow = CFMHead(
            dim_in = len(self.order),
            dim_cond = self.dim_embed,
            dim_time = 64,
            dim_hidden = 64,
        )

        self.optimizer = torch.optim.RAdam(self.parameters(), lr=0.01)


class SurrogateSplit(nn.Module):
    def __init__(
        self,
        shapes,
        means,
        stds,
        max_seq_len,
        feature_dict,
        classification,
        regression,
        multiplicity,
        loss_factors,
        device,
    ):
        super().__init__()

        self.dim_embed = 64
        self.feature_dict = feature_dict
        assert set(self.feature_dict.keys()) == set(classification+regression), f'Mismatch beteen features in inputs ({list(self.feature_dict.keys())}) and their split into classification ({classification}) and regression ({regression})'
        self.teacher_forcing = 1.0
        self.multiplicity = multiplicity
        assert self.multiplicity in ['none','hard','soft']
        self.device = device

        # Prevent logits normalisation #
        for feature in classification:
            for idx in self.feature_dict[feature]:
                means['reco'][idx] = 0.
                stds['reco'][idx] = 1.

        # Parameters #
        self.means = nn.ParameterDict(
            {
                key : nn.Parameter(tensor,requires_grad=False)
                for key,tensor in means.items()
            }
        )
        self.stds = nn.ParameterDict(
            {
                key : nn.Parameter(tensor,requires_grad=False)
                for key,tensor in stds.items()
            }
        )
        self.loss_factors = nn.ParameterDict(
            {
                key : nn.Parameter(torch.tensor(val),requires_grad=False)
                for key,val in loss_factors.items()
            }
        )
        self.max_seq_len = max_seq_len

        # Pre-transformer transformations #
        self.embeddings = nn.ModuleDict(
            {
                name : nn.Sequential(
                    nn.Linear(shape,16),
                    nn.GELU(),
                    nn.Linear(16,32),
                    nn.GELU(),
                    nn.Linear(32,self.dim_embed),
                )
                for name,shape in shapes.items()
            }
        )

        # Transformer #
        self.encoder = Encoder(
            dim_embed = self.dim_embed,
            nhead = 4,
            num_layers = 4,
            expansion_factor = 4,
            dropout = 0.10,
        )
        self.decoder = Decoder(
            dim_embed = self.dim_embed,
            nhead = 4,
            num_layers = 6,
            expansion_factor = 4,
            dropout = 0.10,
            is_causal = True,
        )
        self.start_token = nn.Parameter(
            torch.zeros(1,1,self.dim_embed),
            requires_grad = True,
        )

        # Multiplicity #
        if self.multiplicity == 'hard':
            self.mult_hard_token = nn.Parameter(
                torch.randn(1,1,self.dim_embed),
                requires_grad = True,
            )
            self.mult_hard_encoding = CrossEncoder(
                dim_embed = self.dim_embed,
                nhead = 4,
                num_layers = 3,
                expansion_factor = 2,
                dropout = 0.1,
            )
            self.mult_hard_head = ClassifierMultiplicity(
                dim_embed = self.dim_embed,
                max_seq_len = self.max_seq_len,
                dropout = 0.1,
            )
        if self.multiplicity == 'soft':
            self.mult_soft_head = ClassifierExist(
                dim_embed = self.dim_embed,
                alpha = 0.25,
                gamma = 2,
                max_seq_len = self.max_seq_len,
                dropout = 0.1,
            )

        # Timing #
        if 'time' in shapes.keys():
            self.time_token = nn.Parameter(
                torch.randn(1,1,self.dim_embed),
                requires_grad = True,
            )
            self.time_encoding = CrossEncoder(
                dim_embed = self.dim_embed,
                nhead = 4,
                num_layers = 3,
                expansion_factor = 2,
                dropout = 0.2,
            )
            self.time_head = zuko.flows.NSF(
                features = 1,
                context = self.dim_embed,
                bins = 16,
                transforms = 3,
                randperm = True,
                hidden_features = [128,128],
            )
        else:
            self.time_token = None
            self.time_encoding = None
            self.time_head = None

        # Regression (flow) heads #
        dim_add_context = 0
        self.regression_heads = nn.ModuleDict()
        for feature in regression:
            assert feature in self.feature_dict.keys()
            #flow = zuko.flows.NSF(
            #    features = len(self.feature_dict[feature]),
            #    context = self.dim_cond + dim_add_context,
            #    bins = 8,
            #    transforms = 5,
            #    randperm = True,
            #    passes = None,
            #    hidden_features = [256,256],
            #) # passes = None -> Autoregressive (MAF style)
            ##self.regression_heads[feature] = zuko.flows.Flow(flow.transform.inv, flow.base)
            #self.regression_heads[feature] = flow
            self.regression_heads[feature] = INNHead(len(self.feature_dict[feature]),self.dim_cond + dim_add_context)
            # Turn MAF into IAF (more efficient for sampling)
            dim_add_context += len(self.feature_dict[feature])

        # Classification heads #
        self.classification_heads = nn.ModuleDict()
        for feature in classification:
            assert feature in self.feature_dict.keys()
            self.classification_heads[feature] = ClassifierHead(
                dim_embed = self.dim_cond + dim_add_context,
                n_classes = len(self.feature_dict[feature]),
                dim_encoding = 16,
                dropout = 0.10,
            )
            dim_add_context += self.classification_heads[feature].dim_encoding

        # Miscellanious #
        self.optimizer = torch.optim.RAdam(self.parameters(), lr=0.01)
        self.surrogate_loss = []

    def set_to_device(self,device):
        self.device = device
        self.to(self.device)

    def apply_preprocessing(self, batch):
        # instead of replacing the original tensors
        x_true = (batch['true']['data'] - self.means['true']) / self.stds['true']
        x_reco = (batch['reco']['data'] - self.means['reco']) / self.stds['reco']
        x_params = (batch['params'] - self.means['params']) / self.stds['params']
        batch_new = {
            'true': {'data': x_true, 'mask': batch['true']['mask']},
            'reco': {'data': x_reco, 'mask': batch['reco']['mask']},
            'params': x_params,
        }
        if 'time' in batch.keys():
            batch_new['time'] = (batch['time'] - self.means['time']) / self.stds['time']
        return batch_new

    def undo_preprocessing(self, batch):
        # instead of replacing the original tensors
        x_true = (batch['true']['data'] * self.stds['true']) + self.means['true']
        x_reco = (batch['reco']['data'] * self.stds['reco']) + self.means['reco']
        x_params = (batch['params'] * self.stds['params']) + self.means['params']
        batch_new = {
            'true': {'data': x_true, 'mask': batch['true']['mask']},
            'reco': {'data': x_reco, 'mask': batch['reco']['mask']},
            'params': x_params,
        }
        if 'time' in batch.keys():
            batch_new['time'] = (batch['time'] * self.stds['time']) + self.means['time']
        return batch_new


    def apply_embeddings(self,batch):
        x_enc = torch.cat(
            [
                self.embeddings['params'](
                    batch['params']
                ),
                self.embeddings['true'](
                    batch['true']['data']
                ),
            ],
            dim = 1,
        )
        m_enc = torch.cat(
            [
                torch.full(
                    (batch['true']['mask'].shape[0],1),
                    fill_value = True,
                ).to(batch['true']['mask'].device),
                batch['true']['mask'],
            ],
            dim = 1,
        )
        x_dec = torch.cat(
            [
                self.start_token.expand(batch['reco']['data'].shape[0], -1, -1),
                self.embeddings['reco'](
                    batch['reco']['data'][:,:-1]
                ),
            ],
            dim = 1,
        )
        m_dec = torch.cat(
            [
                torch.full(
                    (batch['reco']['mask'].shape[0],1),
                    fill_value = True,
                ).to(batch['reco']['mask'].device),
                batch['reco']['mask'][:,:-1],
            ],
            dim = 1,
        )
        return x_enc,m_enc,x_dec,m_dec

    def forward(self,batch):
        # Preprocessing #
        batch = self.apply_preprocessing(batch)
        # Select (preprocessed) targets #
        reco_target = batch['reco']['data'].clone()
        mask_target = batch['reco']['mask'].clone()
#        # Sample batch for scheduled sampling / Exposure bias mitigation #
#        if self.teacher_forcing < 1:
#            probs = torch.rand(mask_target.shape).to(mask_target.device)
#            probs[~mask_target] = 1
#            replace_mask = probs < (1-self.teacher_forcing)
#            in_training = self.training
#            self.eval()
#            with torch.no_grad():
#                samples,_,_ = self._sample(batch,replace_mask)
#            if in_training:
#                self.train()
#            #batch['reco']['data'] = samples
#            #replace_mask = torch.logical_and(
#            #    self.log_prob(self.undo_preprocessing(batch))['pos'] < -3.,
#            #    batch['reco']['mask'],
#            #)
#            #batch['reco']['data'][replace_mask] = samples[replace_mask]
#            #print ('replace',replace_mask.sum(),replace_mask.sum()/batch['reco']['mask'].sum())
        # Embedding #
        x_enc,m_enc,x_dec,m_dec = self.apply_embeddings(batch)
        # Pass through transformer #
        x_enc = self.encoder(
            x = x_enc,
            mask = ~m_enc,
        )
        condition = self.decoder(
            x = x_dec,
            mask = ~m_dec,
            memory = x_enc,
            memory_mask = ~m_enc,
        )
        # Multiplicity classifiers #
        if self.multiplicity == 'hard':
            mult_enc = self.mult_hard_encoding(
                x = x_enc,
                mask = ~m_enc,
            ).squeeze(dim=1)
            y = self.mult_hard_head(mult_enc)
            mult_losses = self.mult_hard_head.loss(y,mask_target)
        elif self.multiplicity == 'soft':
            y = self.mult_soft_head(condition)
            mult_losses = self.mult_soft_head.loss(y.squeeze(-1),mask_target)
        else:
            mult_losses = None

        # Pass through regression head / flow #
        reg_losses = {}
        for feature in self.regression_heads.keys():
            t = torch.index_select(
                input = reco_target,
                dim = -1,
                index = self.feature_dict[feature].to(reco_target.device),
            )
            #reg_losses[feature] = - (
            #    self.regression_heads[feature](condition).log_prob(t) * mask_target
            #).sum(dim=-1)
            reg_losses[feature] = self.regression_heads[feature].loss(condition,t,mask_target)
            condition = torch.cat([condition,t],dim=-1)

        # Pass through classifier heads #
        class_losses = {}
        for feature in self.classification_heads.keys():
            t = torch.index_select(
                input = reco_target,
                dim = -1,
                index = self.feature_dict[feature].to(reco_target.device),
            )
            y = self.classification_heads[feature](condition)
            class_losses[feature] = self.classification_heads[feature].loss(y,mask_target,t)
            condition = torch.cat(
                [
                    condition,
                    self.classification_heads[feature].encode(y)
                ],
                dim = -1
            )

        # Process time flow #
        if 'time' in batch.keys():
            time_enc = self.time_encoding(
                x = x_enc,
                mask = ~m_enc,
            ).squeeze(dim=1)
            #time_enc = F.dropout(time_enc, p=0.1, training=self.training)
            time_losses = - self.time_head(time_enc).log_prob(batch['time'])
        else:
            time_losses = None

        return class_losses, reg_losses, mult_losses, time_losses

    @torch.no_grad()
    def log_prob(self,batch):
        # Preprocessing #
        batch = self.apply_preprocessing(batch)
        # Select (preprocessed) targets #
        reco_target = batch['reco']['data'].clone()
        mask_target = batch['reco']['mask'].clone()
        # Embedding #
        x_enc,m_enc,x_dec,m_dec = self.apply_embeddings(batch)
        # Pass through transformer #
        x_enc = self.encoder(
            x = x_enc,
            mask = ~m_enc,
        )
        condition = self.decoder(
            x = x_dec,
            mask = ~m_dec,
            memory = x_enc,
            memory_mask = ~m_enc,
        )
        condition = self.projection(condition)
        log_probs = {}
        # Pass through regression head / flow #
        for feature in self.regression_heads.keys():
            t = torch.index_select(
                input = reco_target,
                dim = -1,
                index = self.feature_dict[feature].to(reco_target.device),
            )
            log_probs[feature] = self.regression_heads[feature](condition).log_prob(t)
            condition = torch.cat([condition,t],dim=-1)

        # Pass through classifier heads #
        for feature in self.classification_heads.keys():
            t = torch.index_select(
                input = reco_target,
                dim = -1,
                index = self.feature_dict[feature].to(reco_target.device),
            )
            y = self.classification_heads[feature](condition)
            log_probs[feature] = self.classification_heads[feature].loss(y,mask_target,t)
            condition = torch.cat(
                [
                    condition,
                    self.classification_heads[feature].encode(y)
                ],
                dim = -1
            )
        return log_probs

    def encode(self,batch):
        # Preprocessing #
        batch = self.apply_preprocessing(batch)
        # Embedding #
        x_enc,m_enc,x_dec,m_dec = self.apply_embeddings(batch)
        # Pass through transformer #
        x_enc = self.encoder(
            x = x_enc,
            mask = ~m_enc,
        )
        condition = self.decoder(
            x = x_dec,
            mask = ~m_dec,
            memory = x_enc,
            memory_mask = ~m_enc,
        )
        condition = self.projection(condition)
        return condition


    @staticmethod
    def move_batch(batch,device):
        batch_new = {
            'true': {
                'data': batch['true']['data'].to(device),
                'mask': batch['true']['mask'].to(device),
            },
            'reco': {
                'data': batch['reco']['data'].to(device),
                'mask': batch['reco']['mask'].to(device),
            },
            'params' : batch['params'].to(device),
        }
        if 'time' in batch.keys():
            batch_new['time'] = batch['time'].to(device)
        return batch_new

    def train_model(
        self,
        train_dataset: SurrogateDataset,
        valid_dataset: SurrogateDataset = None,
        batch_size: int = 64,
        n_epochs: int = 1,
        lr: float = 1e-3,
        teacher_forcing: float = 1.0,
        plotter: LossPlotting = None,
        loss_matching = None,
        num_workers = None,
    ):
        train_loader = DataLoader(
            dataset = train_dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers = num_workers,
            pin_memory = True,
        )
        if valid_dataset is None:
            valid_loader = None
        else:
            valid_loader = DataLoader(
                dataset = valid_dataset,
                batch_size = batch_size,
                shuffle = True,
                num_workers = num_workers,
                pin_memory = True,
            )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.to(self.device)

        assert teacher_forcing >= 0 and teacher_forcing <= 1
        print(f"Surrogate Training: {n_epochs=}, {lr=}, {batch_size=}, {teacher_forcing=}")
        self.teacher_forcing = teacher_forcing
        for key,values in self.loss_factors.items():
            print (f'{key:15s} : {values.item():6.3f}')

        for epoch in range(n_epochs):
            # Training #
            self.train()
            train_losses = {
                'total' : torch.zeros(len(train_loader)),
                **{
                    name : torch.zeros(len(train_loader))
                    for name in self.regression_heads.keys()
                },
                **{
                    name : torch.zeros(len(train_loader))
                    for name in self.classification_heads.keys()
                },
            }
            if self.multiplicity != 'none':
                train_losses['mult'] = torch.zeros(len(train_loader))
            if self.time_head is not None:
                train_losses['time'] = torch.zeros(len(train_loader))
            if loss_matching is not None:
                train_losses['matching'] = torch.zeros(len(train_loader))
            for batch_idx, batch in tqdm(enumerate(train_loader),total=len(train_loader),desc='Training batches',leave=False):
                # Move to device #
                batch = self.move_batch(batch,self.device)
                # Process #
                class_losses,reg_losses,mult_losses,time_losses = self(batch)
                # Record #
                tot_losses = []
                for name,losses in reg_losses.items():
                    train_losses[name][batch_idx] = losses.mean().item()
                    if name in self.loss_factors.keys():
                        factor = self.loss_factors[name]
                    else:
                        factor = 1.
                    tot_losses.append(factor * losses)
                for name,losses in class_losses.items():
                    train_losses[name][batch_idx] = losses.mean().item()
                    if name in self.loss_factors.keys():
                        factor = self.loss_factors[name]
                    else:
                        factor = 1.
                    tot_losses.append(factor * losses)
                if time_losses is not None:
                    train_losses['time'][batch_idx] = time_losses.mean()
                    if 'time' in self.loss_factors.keys():
                        factor = self.loss_factors['time']
                    else:
                        factor = 1.
                    tot_losses.append(factor * time_losses)
                if mult_losses is not None:
                    train_losses['mult'][batch_idx] = mult_losses.mean()
                    if 'mult' in self.loss_factors.keys():
                        factor = self.loss_factors['mult']
                    else:
                        factor = 1.
                    tot_losses.append(factor * mult_losses)
                if loss_matching is not None:
                    self.eval()
                    samples,samples_mask,_ = self.sample(batch)
                    self.train()
                    loss_match_values = loss_matching(
                        batch['reco']['data'],
                        batch['reco']['mask'],
                        samples,
                        samples_mask,
                    )['total']
                    train_losses['matching'][batch_idx] = loss_match_values.mean()
                    if 'matching' in self.loss_factors.keys():
                        factor = self.loss_factors['matching']
                    else:
                        factor = 1.
                    tot_losses.append(factor * loss_match_values)
                tot_loss = sum(tot_losses).mean()
                train_losses['total'][batch_idx] = tot_loss.item()
                # Optimizer step #
                self.optimizer.zero_grad()
                tot_loss.backward()
                self.optimizer.step()
            # Validation #
            self.eval()
            if valid_loader is not None:
                valid_losses = {
                    'total' : torch.zeros(len(valid_loader)),
                    **{
                        name : torch.zeros(len(valid_loader))
                        for name in self.regression_heads.keys()
                    },
                    **{
                        name : torch.zeros(len(valid_loader))
                        for name in self.classification_heads.keys()
                    },
                }
                if self.time_head is not None:
                    valid_losses['time'] = torch.zeros(len(valid_loader))
                if self.multiplicity != 'none':
                    valid_losses['mult'] = torch.zeros(len(valid_loader))
                if loss_matching is not None:
                    valid_losses['matching'] = torch.zeros(len(valid_loader))
                for batch_idx, batch in tqdm(enumerate(valid_loader),total=len(valid_loader),desc='Validation batches',leave=False):
                    # Move to device #
                    batch = self.move_batch(batch,self.device)
                    # Process #
                    with torch.no_grad():
                        class_losses,reg_losses,mult_losses,time_losses = self(batch)
                    # Record #
                    tot_losses = []
                    for name,losses in reg_losses.items():
                        valid_losses[name][batch_idx] = losses.mean().item()
                        valid_losses[name][batch_idx] = losses.mean().item()
                        if name in self.loss_factors.keys():
                            factor = self.loss_factors[name]
                        else:
                            factor = 1.
                        tot_losses.append(factor * losses)
                    for name,losses in class_losses.items():
                        valid_losses[name][batch_idx] = losses.mean().item()
                        if name in self.loss_factors.keys():
                            factor = self.loss_factors[name]
                        else:
                            factor = 1.
                        tot_losses.append(factor * losses)
                    if time_losses is not None:
                        valid_losses['time'][batch_idx] = time_losses.mean()
                        if 'time' in self.loss_factors.keys():
                            factor = self.loss_factors['time']
                        else:
                            factor = 1.
                        tot_losses.append(factor * time_losses)
                    if mult_losses is not None:
                        valid_losses['mult'][batch_idx] = mult_losses.mean()
                        if 'mult' in self.loss_factors.keys():
                            factor = self.loss_factors['mult']
                        else:
                            factor = 1.
                        tot_losses.append(factor * mult_losses)
                    if loss_matching is not None:
                        with torch.no_grad():
                            samples,samples_mask,_ = self.sample(batch)
                        loss_match_values = loss_matching(
                            batch['reco']['data'],
                            batch['reco']['mask'],
                            samples,
                            samples_mask,
                        )['total']
                        valid_losses['matching'][batch_idx] = loss_match_values.mean()
                        if 'matching' in self.loss_factors.keys():
                            factor = self.loss_factors['matching']
                        else:
                            factor = 1.
                        tot_losses.append(factor * loss_match_values)
                    tot_loss = sum(tot_losses).mean()
                    valid_losses['total'][batch_idx] = tot_loss.item()
            else:
                valid_losses = {
                    'total' : torch.zeros(1),
                    **{
                        name : torch.zeros(1)
                        for name in self.classification_heads.keys()
                    },
                    **{
                        name : torch.zeros(1)
                        for name in self.regression_heads.keys()
                    },
                }
                if self.time_head is not None:
                    valid_losses['time'] = torch.zeros(1)
                if self.multiplicity != 'none':
                    valid_losses['mult'] = torch.zeros(1)

            # Printout #
            s = f"Surrogate epoch = {epoch:3d} - Loss = {train_losses['total'].mean():+10.5f} [{valid_losses['total'].mean():+10.5f}] : "
            for name in train_losses.keys():
                if name == 'total':
                    continue
                if name in self.loss_factors.keys():
                    factor = self.loss_factors[name]
                else:
                    factor = 1.
                s += f"{name} = {factor.item(): .3f} x {train_losses[name].mean():+10.5f} [{valid_losses[name].mean():+10.5f}] + "
            print (s)
            self.surrogate_loss.append(valid_losses['total'].mean())

            # Plotter #
            if plotter is not None:
                plotter.add_lr_value(lr)
                for name,losses in train_losses.items():
                    plotter.add_train_value(name,losses.mean().item())
                for name,losses in valid_losses.items():
                    plotter.add_valid_value(name,losses.mean().item())

    def _sample_regression_head(self,feature,cond,threshold):
        flow = self.regression_heads[feature]
        y = torch.zeros((cond.shape[0],len(self.feature_dict[feature])),device=cond.device)
        log_probs = torch.full((cond.shape[0],),fill_value=-torch.inf,device=cond.device)
        while (log_probs < threshold).sum() > 0:
            idx = log_probs < threshold
            y[idx] = flow(cond[idx]).rsample()
            log_probs[idx] = flow(cond[idx]).log_prob(y[idx])
        return y

    def _sample(self,batch,thresholds,replace_mask=None):
        """ Sample without preprocessing """
        # Defaut preprocessing #
        if thresholds is None:
            thresholds = {}
        for name in self.regression_heads.keys():
            if name not in thresholds.keys():
                thresholds[name] = -math.inf

        # Get targets #
        reco_target = batch['reco']['data'].clone()
        mask_target = batch['reco']['mask'].clone()
        # Embedding #
        x_enc, m_enc, x_dec, m_dec = self.apply_embeddings(batch)
        # Pass through encoder #
        x_enc = self.encoder(
            x = x_enc,
            mask = ~m_enc,
        )
        # Timing #
        if self.time_head is not None:
            time_enc = self.time_encoding(
                slots = self.time_token.expand(x_enc.shape[0], -1, -1),
                encoder_out = x_enc,
                encoder_padding_mask = ~m_enc,
            ).squeeze(dim=1)
            times = self.time_head(time_enc).rsample()
            times = times * self.stds['time'] + self.means['time']
        else:
            times = None
        # Make empty decoder input (with start token) #
        if self.multiplicity =='hard' and replace_mask is None:
            # Hard classifier : we sample the sampled mask, and replace decoder mask
            mult_enc = self.mult_hard_encoding(
                slots = self.mult_hard_token.expand(x_enc.shape[0], -1, -1),
                encoder_out = x_enc,
                encoder_padding_mask = ~m_enc,
            ).squeeze(dim=1)
            samples_mask = self.mult_hard_head.sample(mult_enc)
            m_dec = torch.cat(
                [
                    torch.full( # add start token mask
                        (samples_mask.shape[0],1),
                        fill_value = True,
                    ).to(mask_target.device),
                    samples_mask,
                ],
                dim = 1,
            )
            N = self.max_seq_len
        elif self.multiplicity == 'soft' and replace_mask is None:
            # Soft classifier : we use unmasked particles and predict later
            m_dec = torch.full(
                (
                    m_dec.shape[0],
                    self.max_seq_len,
                ),
                fill_value = True,
            ).to(m_dec.device)
            N = self.max_seq_len
        else:
            samples_mask = mask_target
            N = mask_target.shape[1]
        samples = []
        x_dec = [x_dec[:,0:1,:]] # only keep start token
        # Sequential generation #
        for i in range(N):
            # Pass through decoder, obtain new context #
            condition = self.decoder(
                x = torch.cat(x_dec,dim=1),
                mask = ~m_dec[:,:len(x_dec)],
                memory = x_enc,
                memory_mask = ~m_enc,
            )[:,i,:]
            condition = self.projection(condition)

            # Create placeholder particle
            particle = torch.zeros(
                (
                    batch['reco']['data'].shape[0],
                    batch['reco']['data'].shape[2],
                )
            ).to(self.device)
            # Regression heads #
            for feature in self.regression_heads.keys():
                y = self._sample_regression_head(
                    feature = feature,
                    cond = condition,
                    threshold = thresholds[feature],
                )
                idx = self.feature_dict[feature].to(y.device)
                particle = particle.scatter(-1, idx.view(1, -1).expand(y.shape[0], -1) ,y) # keep grads
                condition = torch.cat([condition,y],dim=-1)
            # Classification heads #
            for feature in self.classification_heads.keys():
                y = self.classification_heads[feature](condition)
                idx = self.feature_dict[feature].to(y.device)
                particle = particle.scatter(-1, idx.view(1, -1).expand(y.shape[0], -1) ,y) # keep grads
                condition = torch.cat(
                    [
                        condition,
                        self.classification_heads[feature].encode(y)
                    ],
                    dim = -1,
                )
            # Add to samples and pass through embedding for new token #
            if replace_mask is not None:
                particle[~replace_mask[:,i]] = reco_target[:,i,:][~replace_mask[:,i]]
            samples.append(particle)
            x_dec.append(self.embeddings['reco'](particle).unsqueeze(1))
        # Soft classifier for existence sampling #
        if self.multiplicity == 'soft' and replace_mask is None:
            condition = self.decoder(
                x = torch.cat(x_dec[:-1],dim=1),
                mask = ~m_dec,
                memory = x_enc,
                memory_mask = ~m_enc,
            )
            samples_mask = self.mult_soft_head.sample(condition)
        # Stack and zero-out missing particles #
        samples = torch.stack(samples,dim=1)
        samples = (samples * samples_mask.unsqueeze(-1))
        # Sort so that True particles (present) are before False (absent)
        idx = torch.argsort(samples_mask.int(),dim=1,descending=True,stable=True)
        samples = torch.gather(
            input = samples,
            dim = 1,
            index = idx.unsqueeze(-1).expand(-1, -1, samples.size(-1)),
        )
        samples_mask = samples_mask[
            torch.arange(samples_mask.size(0), device=samples_mask.device)[:, None],
            idx,
        ]

        return samples,samples_mask,times


    def sample(self,batch,thresholds):
        # Preprocessing #
        batch = self.apply_preprocessing(batch)
        # Sampling #
        samples, samples_mask, times = self._sample(batch,thresholds)
        # Undo preprocessing #
        samples = samples * self.stds['reco'] + self.means['reco']
        return samples,samples_mask,times

    @torch.no_grad()
    def inference(
        self,
        dataset,
        batch_size,
        thresholds,
        oversampling: int = 1,
    ):
        self.eval()
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        if self.multiplicity == 'none':
            max_seq_len = dataset[0]['reco']['data'].shape[0]
        else:
            max_seq_len = self.max_seq_len
        n_features  = sum(
            [
                len(idxs)
                for idxs in self.feature_dict.values()
            ]
        )
        all_samples = torch.zeros((oversampling, len(dataset), max_seq_len, n_features))
        all_masks = torch.zeros((oversampling, len(dataset), max_seq_len)) > 0
        all_times = torch.zeros((oversampling, len(dataset), 1))

        for io in tqdm(range(oversampling),desc='Oversample',total=oversampling,leave=True,position=0):

            for batch_idx, batch in tqdm(enumerate(data_loader),desc='Sample',total=len(data_loader),leave=False,position=1):
                # Move to device #
                batch = self.move_batch(batch,self.device)
                # Sample #
                samples,mask,times = self.sample(batch,thresholds)
                # Record #
                ia = batch_idx * batch_size
                ib = (batch_idx + 1) * batch_size

                all_samples[io,ia:ib] = samples.cpu()
                all_masks[io,ia:ib] = mask.cpu()
                if times is not None:
                    all_times[io,ia:ib] = times.cpu()

        return all_samples,all_masks,all_times

    def compute_log_prob_thresholds(self,dataset,batch_size,quantile):
        self.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_log_probs = {}
        for batch in tqdm(loader,total=len(loader),desc='Batches'):
            batch = self.move_batch(batch, self.device)
            log_probs = self.log_prob(batch)
            for key,values in log_probs.items():
                if key in all_log_probs.keys():
                    all_log_probs[key].append(values)
                else:
                    all_log_probs[key] = [values]
        thresholds = {}
        for key,log_probs in all_log_probs.items():
            thresholds[key] = torch.quantile(torch.cat(log_probs,dim=0), quantile).item()
        return thresholds

class AttentionExtractor:
    def __init__(self):
        self.hooks = []
        self._patched = []
        self.self_attn = {}    # layer_idx -> list of (B, heads, tgt_len, tgt_len)
        self.cross_attn = {}   # layer_idx -> list of (B, heads, tgt_len, src_len)

    def register(self, decoder: nn.TransformerDecoder):
        for i, layer in enumerate(decoder.layers):
            layer.self_attn.average_attn_weights = False
            layer.multihead_attn.average_attn_weights = False

            self._patch_mha(layer.self_attn)
            self._patch_mha(layer.multihead_attn)

            self.hooks.append(
                layer.self_attn.register_forward_hook(
                    self._make_hook(self.self_attn, i)
                )
            )
            self.hooks.append(
                layer.multihead_attn.register_forward_hook(
                    self._make_hook(self.cross_attn, i)
                )
            )

    def _patch_mha(self, mha: nn.MultiheadAttention):
        original_forward = mha.forward
        def patched_forward(*args, **kwargs):
            kwargs['need_weights'] = True
            kwargs['average_attn_weights'] = False
            return original_forward(*args, **kwargs)
        mha.forward = patched_forward
        self._patched.append((mha, original_forward))

    def _make_hook(self, store, layer_idx):
        def hook(module, input, output):
            # output = (attn_output, attn_weights)
            # attn_weights: (B, heads, tgt_len, src_len) or None
            if output[1] is not None:
                store[layer_idx] = output[1].detach().cpu()
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()
        for mha, original in self._patched:
            mha.forward = original
        self.hooks = []
        self._patched = []

#    def plot_cross_attn(self, sample_idx, layer_idx=-1, figsize=(10, 4)):
#        """
#        Plot cross-attention map for a single sample.
#        Rows = decoder tokens, Cols = encoder tokens.
#        Each head is shown separately, plus the mean over heads.
#        """
#        # Normalise layer index
#        keys = sorted(self.cross_attn.keys())
#        layer_key = keys[layer_idx]
#
#        attn = self.cross_attn[layer_key]  # (B, heads, tgt_len, src_len)
#        attn = attn[sample_idx]            # (heads, tgt_len, src_len)
#        n_heads = attn.shape[0]
#
#        fig, axes = plt.subplots(
#            1, n_heads + 1,
#            figsize=(figsize[0] * (n_heads + 1) / 2, figsize[1])
#        )
#
#        for h in range(n_heads):
#            axes[h].imshow(attn[h].numpy(), aspect='auto', cmap='viridis')
#            axes[h].set_title(f'Head {h}')
#            axes[h].set_xlabel('Encoder token')
#            axes[h].set_ylabel('Decoder token')
#
#        # Mean over heads
#        axes[-1].imshow(attn.mean(0).numpy(), aspect='auto', cmap='viridis')
#        axes[-1].set_title('Mean over heads')
#        axes[-1].set_xlabel('Encoder token')
#        axes[-1].set_ylabel('Decoder token')
#
#        plt.suptitle(f'Cross-attention — layer {layer_key} — sample {sample_idx}')
#        plt.tight_layout()
#        return fig
#
#    def plot_self_attn(self, sample_idx, layer_idx=-1, figsize=(10, 4)):
#        """
#        Plot decoder self-attention map for a single sample.
#        Rows = query tokens, Cols = key tokens.
#        Each head is shown separately, plus the mean over heads.
#        """
#        keys = sorted(self.self_attn.keys())
#        layer_key = keys[layer_idx]
#
#        attn = self.self_attn[layer_key]   # (B, heads, tgt_len, tgt_len)
#        attn = attn[sample_idx]            # (heads, tgt_len, tgt_len)
#        n_heads = attn.shape[0]
#
#        fig, axes = plt.subplots(
#            1, n_heads + 1,
#            figsize=(figsize[0] * (n_heads + 1) / 2, figsize[1])
#        )
#
#        for h in range(n_heads):
#            axes[h].imshow(attn[h].numpy(), aspect='auto', cmap='viridis')
#            axes[h].set_title(f'Head {h}')
#            axes[h].set_xlabel('Key token')
#            axes[h].set_ylabel('Query token')
#
#        axes[-1].imshow(attn.mean(0).numpy(), aspect='auto', cmap='viridis')
#        axes[-1].set_title('Mean over heads')
#        axes[-1].set_xlabel('Key token')
#        axes[-1].set_ylabel('Query token')
#
#        plt.suptitle(f'Self-attention — layer {layer_key} — sample {sample_idx}')
#        plt.tight_layout()
#        return fig
#
#    def check_double_consumption(self, sample_idx, layer_idx=-1, threshold=0.3):
#        """
#        For a given sample, check if any two decoder tokens dominantly
#        attend to the same encoder token (double consumption).
#        threshold: minimum attention weight to consider 'dominant'
#        """
#        keys = sorted(self.cross_attn.keys())
#        layer_key = keys[layer_idx]
#
#        attn = self.cross_attn[layer_key]  # (B, heads, tgt_len, src_len)
#        attn = attn[sample_idx].mean(0)    # (tgt_len, src_len) — mean over heads
#
#        dominant = attn.argmax(dim=-1)     # (tgt_len,) — which encoder token each decoder step attends to most
#
#        seen = {}
#        issues = []
#        for tgt_idx, src_idx in enumerate(dominant.tolist()):
#            if attn[tgt_idx, src_idx] < threshold:
#                continue  # attention is diffuse, skip
#            if src_idx in seen:
#                issues.append((tgt_idx, src_idx, seen[src_idx]))
#                print(f"  Double consumption: decoder token {tgt_idx} and {seen[src_idx]} "
#                      f"both attend to encoder token {src_idx} "
#                      f"(attn={attn[tgt_idx, src_idx]:.2f})")
#            else:
#                seen[src_idx] = tgt_idx
#
#        if not issues:
#            print(f"  No double consumption detected (threshold={threshold})")
#
#        return issues

def validation_plot(
    loss_matching,
    true_part,
    true_mask,
    reco_part,
    reco_mask,
    samp_part,
    samp_mask,
    reco_time = None,
    samp_time = None,
    savepath = None,
    show = False,
):
    # Matching loss plot #
    print ('Computing validation matching loss on reco particles')
    reco_losses = loss_matching(
        true_part,
        true_mask,
        reco_part,
        reco_mask,
    )
    print ('Computing validation matching loss on sampled particles')
    sample_losses = [
        loss_matching(
            true_part,
            true_mask,
            samp_part[i],
            samp_mask[i],
        )
        for i in range(samp_part.shape[0])
    ]

    ncols = len(reco_losses) + 1
    if samp_time is not None:
        ncols += 1
    fig,axs = plt.subplots(ncols=ncols,figsize=(6*ncols,5))
    plt.subplots_adjust(left=0.05,right=0.95,top=0.9,bottom=0.1,wspace=0.3)

    for i,name in enumerate(reco_losses.keys()):
        reco_loss = reco_losses[name].ravel().repeat(len(sample_losses))
        sample_loss = torch.cat(
            [
                samp_losses[name]
                for samp_losses in sample_losses
            ]
        ).ravel()
        min_loss = min(reco_loss.min(),sample_loss.min())
        max_loss = max(reco_loss.max(),sample_loss.max())
        if torch.all(reco_loss == reco_loss.floor()):
            bins = torch.arange(min_loss,max_loss+1)
        else:
            #bins = np.logspace(np.log10(min_loss),np.log10(max_loss),51)
            bins = np.linspace(min_loss,max_loss,51)
            axs[i].plot(
                [min_loss,max_loss],
                [min_loss,max_loss],
                linestyle = '--',
                color = 'grey',
                linewidth = 2,
            )
            #axs[i].set_xscale('log')
            #axs[i].set_yscale('log')
        H = axs[i].hist2d(
            reco_loss.numpy(),
            sample_loss.numpy(),
            bins = bins,
            norm = matplotlib.colors.LogNorm(),
        )
        cbar = plt.colorbar(H[3],ax=axs[i])
        axs[i].set_xlabel(f'Reco {name} matching loss',fontsize=16)
        axs[i].set_ylabel(f'Surrogate {name} matching loss',fontsize=16)
        axs[i].set_aspect('equal')

    # Confusion matrix for multiplicity #
    N_reco = reco_mask.sum(dim=-1).repeat(samp_mask.shape[0])
    N_samp = samp_mask.sum(dim=-1).ravel()
    cm = confusion_matrix(N_samp, N_reco, labels=np.arange(reco_mask.shape[-1]))

    disp = ConfusionMatrixDisplay(
        confusion_matrix = cm,
        display_labels = np.arange(reco_mask.shape[-1]),
    )
    i = len(reco_losses.keys())
    colors = plt.cm.rainbow(np.linspace(0,1,true_mask.shape[1]))
    bins = np.arange(reco_mask.shape[-1])
    for j in range(true_mask.shape[1]):
        if (true_mask.sum(dim=-1)==j).sum() > 0:
            axs[i].hist(
                reco_mask[true_mask.sum(dim=-1)==j].sum(dim=-1),
                bins = bins,
                color = colors[j],
                histtype = 'step',
                linestyle = 'solid',
                label = f'N(true) = {j}',
            )
            axs[i].hist(
                samp_mask[torch.arange(samp_mask.shape[0]),true_mask.sum(dim=-1)==j].sum(dim=-1),
                bins = bins,
                color = colors[j],
                histtype = 'step',
                linestyle = 'dotted',
                linewidth = 3,
            )
    axs[i].set_xlabel('Number of reco particles',fontsize=16)
    axs[i].set_ylabel('Frequency',fontsize=16)
    axs[i].set_yscale('log')
    axs[i].legend(loc='upper right',prop={'size': 8})

    #disp.plot(
    #    ax = axs[i],
    #    cmap = 'Blues',
    #    colorbar = True,
    #    values_format = 'd',
    #    text_kw={'fontsize': 8},
    #)
    #axs[i].invert_yaxis()
    #axs[i].set_ylabel('Reco number of particles',fontsize=16)
    #axs[i].set_xlabel('Surrogate number of particles',fontsize=16)

    # Time plot if done #
    if samp_time is not None:
        samp_times = samp_time.squeeze(-1).permute(1,0).clone()
        true_times = reco_time.clone()
        N = math.lcm(samp_times.shape[-1],true_times.shape[-1])
        samp_times = samp_times.repeat_interleave(N // samp_times.shape[-1], dim=1)
        true_times = true_times.repeat_interleave(N // true_times.shape[-1], dim=1)
        true_times = torch.exp(true_times)
        samp_times = torch.exp(samp_times)
        bins = np.linspace(
            min(samp_times.min(),true_times.min()),
            max(samp_times.max(),true_times.max()),
            51
        )
        H = axs[-1].hist2d(
            true_times.ravel().numpy(),
            samp_times.ravel().numpy(),
            bins = bins,
            norm = matplotlib.colors.LogNorm(),
        )
        cbar = plt.colorbar(H[3],ax=axs[-1])
        axs[-1].set_xlabel('Reco inference time',fontsize=16)
        axs[-1].set_ylabel('Sampled inference time',fontsize=16)


    if savepath is not None:
        fig.savefig(savepath)
        print (f'Validation plot saved as {savepath}')
    if show:
        plt.show()


def test_encoder_position_invariance(dataset,model):
    model = model.cpu()

    loader = DataLoader(dataset,batch_size=32,shuffle=False)
    batch = next(iter(loader))

    with torch.no_grad():
        cls_loss_1,reg_loss_1,_,_ = model(batch)

    x = batch['true']['data'].clone()
    mask = batch['true']['mask'].clone()
    B, N = x.shape[:2]
    perm = torch.argsort(torch.rand(B, N), dim=1)
    x_mix = torch.gather(
        x,
        dim = 1,
        index = perm.unsqueeze(-1).expand(-1, -1, x.size(2)),
    )
    mask_mix = torch.gather(mask, 1, perm)

    batch_mix = {
        'params': batch['params'],
        'reco': batch['reco'],
        'true': {
            'data': x_mix,
            'mask': mask_mix
        }
    }

    with torch.no_grad():
        cls_loss_2,reg_loss_2,_,_ = model(batch_mix)

    print ('Class loss')
    print (cls_loss_1)
    print (cls_loss_2)

    print ('Reg loss')
    print (reg_loss_1)
    print (reg_loss_2)



if __name__ == '__main__':
    import sys
    from aido.config import AIDOConfig
    from torch.utils.data import Subset

    config = AIDOConfig()
    dataset = SurrogateDataset.load_sparse(config,sys.argv[1])

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split = int(len(dataset)*0.9)
    train_indices = indices[:split]
    valid_indices = indices[split:]

    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)
    print (len(train_dataset),len(valid_dataset))

    model = Surrogate(
        shapes = dataset.get_shapes(),
        means = dataset.get_means(),
        stds = dataset.get_stds(),
        max_seq_len = dataset.max_seq_len,
        feature_dict = dataset.feature_dict,
        classification = config.surrogate.classification,
        regression = config.surrogate.regression,
        multiplicity = config.surrogate.multiplicity,
        loss_factors = config.surrogate.loss_factors,
    )
    print (model)

    #test_encoder_position_invariance(dataset,model)
    plotter = LossPlotting()

    if torch.cuda.is_available():
        model = model.cuda()

    model.train_model(
        train_dataset,
        valid_dataset,
        n_epochs = 50,
        batch_size = 64,
        lr = 1e-3,
        teacher_forcing = 1.0,
        plotter = plotter,
    )
    model.train_model(
        train_dataset,
        valid_dataset,
        n_epochs = 50,
        batch_size = 64,
        lr = 1e-3,
        teacher_forcing = 0.9,
        plotter = plotter,
    )
    model.train_model(
        train_dataset,
        valid_dataset,
        n_epochs = 50,
        batch_size = 64,
        lr = 1e-4,
        teacher_forcing = 0.75,
        plotter = plotter,
    )
    model.train_model(
        train_dataset,
        valid_dataset,
        n_epochs = 50,
        batch_size = 64,
        lr = 1e-5,
        teacher_forcing = 0.5,
        plotter = plotter,
    )
    plotter.plot('losses.png')

    torch.save(model,'surrogate.pt')
    #model = torch.load('surrogate.pt')

    train_samples, train_samples_mask, train_times = model.inference(train_dataset,batch_size=1024)
    valid_samples, valid_samples_mask, valid_times = model.inference(valid_dataset,batch_size=1024)

    loss_matching = HungarianMatching(
        feature_dict = dataset.feature_dict,
        loss_factors = config.loss.loss_factors,
        fake_penalty = config.loss.fake_penalty,
        missing_penalty = config.loss.missing_penalty,
        classification = config.surrogate.classification,
        regression = config.surrogate.regression,
    )

    validation_plot(
        loss_matching = loss_matching,
        savepath = 'surrogate_train.png',
        true_part = dataset.true_part[train_indices],
        true_mask = dataset.true_mask[train_indices],
        reco_part = dataset.reco_part[train_indices],
        reco_mask = dataset.reco_mask[train_indices],
        samp_part = train_samples,
        samp_mask = train_samples_mask,
        reco_time = dataset.reco_time[train_indices],
        samp_time = train_times,
    )
    validation_plot(
        loss_matching = loss_matching,
        savepath = 'surrogate_valid.png',
        true_part = dataset.true_part[valid_indices],
        true_mask = dataset.true_mask[valid_indices],
        reco_part = dataset.reco_part[valid_indices],
        reco_mask = dataset.reco_mask[valid_indices],
        samp_part = valid_samples,
        samp_mask = valid_samples_mask,
        reco_time = dataset.reco_time[valid_indices],
        samp_time = valid_times,
    )

    dataset.samp_part, dataset.samp_mask, dataset.samp_time = model.inference(dataset,batch_size=1024)
    for i in range(20):
        dataset.plot(config=config,idx=train_indices[i],savepath=f'event_train_{i}.png')
        dataset.plot(config=config,idx=valid_indices[i],savepath=f'event_valid_{i}.png')
    dataset.save_dense(path='dataset_dense.pt')
    #from IPython import embed; embed()
#    dataset.load_dense(path='dataset_dense.pt')
#    for i in range(20):
#        dataset.plot(config=config,idx=i,savepath=f'event_{i}.png')

#
#    for i in range(10):
#        ncols = dataset.reco_part.shape[-1]
#        fig,axs = plt.subplots(ncols=ncols,figsize=(5*ncols,5))
#        if not isinstance(axs,np.ndarray):
#            axs = np.array([axs])
#        for j in range(ncols):
#            common_mask = torch.logical_and(
#                dataset.reco_mask[:,i],
#                dataset.samp_mask[0,:,i],
#            )
#            x = dataset.reco_part[:,i,j][common_mask]
#            y = samples[0,:,i,j][common_mask]
#            axs[j].scatter(
#                x,
#                y
#                s = 1,
#            )
#            minval = min(x.min(),y.min())
#            maxval = max(x.max(),y.max())
#            axs[j].set_xlim(minval,maxval)
#            axs[j].set_ylim(minval,maxval)
#            axs[j].plot([minval,maxval],[minval,maxval],color='grey',linewidth=2)
#        fig.savefig(f'element_{i}.png')
#        plt.close()
#

