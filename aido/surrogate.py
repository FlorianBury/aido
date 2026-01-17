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
import zuko

from aido.logger import logger
from aido.losses import HungarianMatching
from aido.utils import LossPlotting

class SurrogateDataset(Dataset):
    def __init__(
        self,
        feature_dict,
        true_part,
        true_mask,
        reco_part,
        reco_mask,
        params,
        reco_time = None,
        samp_time = None,
        samp_part = None,
        samp_mask = None,
    ):

        self.feature_dict = feature_dict
        self.true_part = true_part
        self.true_mask = true_mask
        self.reco_part = reco_part
        self.reco_mask = reco_mask
        self.params = params
        self.reco_time = reco_time
        self.samp_time = samp_time
        self.samp_part = samp_part
        self.samp_mask = samp_mask

        # Safety checks #
        assert self.true_part.shape[0] == self.reco_part.shape[0]
        assert self.true_part.shape[0] == self.true_mask.shape[0]
        assert self.reco_part.shape[0] == self.reco_mask.shape[0]
        assert self.true_part.shape[0] == self.params.shape[0]
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
            times = data['time']['times']
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
            assert name in data["vertices"].keys()
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

        # Order reco particles #
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

        ## Order true particles #
        #feature_idx = feature_dict[config.surrogate.ordering][0]
        #true_ordering = torch.arange(true_part.shape[1]).unsqueeze(0).repeat_interleave(true_part.shape[0],dim=0)
        #for i in range(true_part.shape[0]):
        #    order = true_part[i,true_mask[i,:]][:,feature_idx].argsort(descending=True)
        #    true_ordering[i,:len(order)] = order

        #true_part = true_part[
        #    torch.arange(true_part.shape[0]).unsqueeze(1),
        #    true_ordering,
        #]
        #true_mask = true_mask[
        #    torch.arange(true_mask.shape[0]).unsqueeze(1),
        #    true_ordering,
        #]

        ## Order reco by matchig with true order #
        #loss_matching = HungarianMatching(
        #    feature_dict = feature_dict,
        #    loss_factors = config.loss.loss_factors,
        #    fake_penalty = config.loss.fake_penalty,
        #    missing_penalty = config.loss.missing_penalty,
        #    classification = config.surrogate.classification,
        #    regression = config.surrogate.regression,
        #)

        #reco_ordering = torch.arange(reco_part.shape[1]).unsqueeze(0).repeat_interleave(reco_part.shape[0],dim=0)
        #for i in range(reco_part.shape[0]):
        #    cost_matrix = loss_matching.pairwise_cost(true_part[i][true_mask[i]],reco_part[i][reco_mask[i]])
        #    row_ind, col_ind = linear_sum_assignment(cost_matrix)
        #    reco_ordering[i,:len(col_ind)] = torch.from_numpy(col_ind[row_ind.argsort()])

        #reco_part = reco_part[
        #    torch.arange(reco_part.shape[0]).unsqueeze(1),
        #    reco_ordering,
        #]
        #reco_mask = reco_mask[
        #    torch.arange(reco_mask.shape[0]).unsqueeze(1),
        #    reco_ordering,
        #]

        return cls(
            feature_dict = feature_dict,
            true_part = true_part,
            true_mask = true_mask,
            reco_part = reco_part,
            reco_mask = reco_mask,
            reco_time = times,
            params = params,
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
    def max_seq_len(self):
        return self.reco_part.shape[1]

    @property
    def n_features(self):
        return self.true_part.shape[2]

    def __getitem__(self,idx):
        event = {
            'true' : {
                'data' : self.true_part[idx],
                'mask' : self.true_mask[idx],
            },
            'reco' : {
                'data' : self.reco_part[idx],
                'mask' : self.reco_mask[idx],
            },
            'params' : self.params[idx]
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

    def plot(self,config,idx,savepath):
        # Safety checks #
        assert 'pos' in self.feature_dict.keys()

        # Make figure and recover event #
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(left=0.05,right=0.7,bottom=0.1,top=0.85)

        true = self.true_part[idx][self.true_mask[idx]]
        reco = self.reco_part[idx][self.reco_mask[idx]]
        if self.samp_part is not None:
            samp = self.samp_part[:,idx][self.samp_mask[:,idx]]
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

        # Plot #
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

        xs = [1.05,1.30,1.55]
        y = 1.0

        base_box = dict(
            y = y + 0.05,
            transform = ax.transAxes,
            va = "center",
            ha = "center",
            fontsize = 18,
            family = "monospace",
            color = 'black',
            bbox = dict(
                facecolor = "none",
                edgecolor = "none",
            ),
        )
        ax.text(
            x = xs[0]+0.075,
            s = 'True',
            **base_box,
        )
        ax.text(
            x = xs[1]+0.075,
            s = 'Reco',
            **base_box,
        )
        ax.text(
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
                transform = ax.transAxes,
                va = "top",
                ha = "left",
                fontsize = 8,
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
                ax.scatter(
                    true[a,idx_pos[0]],
                    true[a,idx_pos[1]],
                    marker = 'X',
                    s = 100,
                    edgecolor = 'black',
                    facecolor = colors[i],
                )
                box_txt = format_info(true[a])
                height = max(height,box_txt.count('\n')*0.05)
                ax.text(
                    x = xs[0],
                    s = box_txt,
                    **base_box,
                )
                min_pos = min(min_pos,true[a,idx_pos].min())
                max_pos = max(max_pos,true[a,idx_pos].max())
            if b is not None:
                ax.scatter(
                    reco[b,idx_pos[0]],
                    reco[b,idx_pos[1]],
                    marker = 'o',
                    s = 100,
                    edgecolor = 'black',
                    facecolor = colors[i],
                )
                box_txt = format_info(reco[b])
                height = max(height,box_txt.count('\n')*0.05)
                ax.text(
                    x = xs[1],
                    s = box_txt,
                    **base_box,
                )
                min_pos = min(min_pos,reco[b,idx_pos].min())
                max_pos = max(max_pos,reco[b,idx_pos].max())
            if samp is not None and groups[i][2] is not None:
                c = groups[i][2]
                ax.scatter(
                    samp[c,idx_pos[0]],
                    samp[c,idx_pos[1]],
                    marker = 'v',
                    s = 100,
                    edgecolor = 'black',
                    facecolor = colors[i],
                )
                box_txt = format_info(samp[c])
                height = max(height,box_txt.count('\n')*0.05)
                ax.text(
                    x = xs[2],
                    s = box_txt,
                    **base_box,
                )
                min_pos = min(min_pos,samp[c,idx_pos].min())
                max_pos = max(max_pos,samp[c,idx_pos].max())
            y -= (height + 0.02)

        # Dummy plot for legend #
        ax.scatter(
            [],[],
            marker = 'X',
            s = 100,
            edgecolor = 'black',
            facecolor = 'white',
            label = 'True',
        )
        ax.scatter(
            [],[],
            marker = 'o',
            s = 100,
            edgecolor = 'black',
            facecolor = 'white',
            label = 'Reco',
        )
        if samp is not None:
            ax.scatter(
                [],[],
                marker = 'v',
                s = 100,
                edgecolor = 'black',
                facecolor = 'white',
                label = 'Surrogate',
            )
        ax.legend(frameon=False,fontsize=18)

        # Esthetics #
        ax.set_xlim(min_pos-5,max_pos+5)
        ax.set_ylim(min_pos-5,max_pos+5)
        ax.set_xlabel('x [cm]',fontsize=18)
        ax.set_ylabel('y [cm]',fontsize=18)
        ax.set_aspect("equal", adjustable="box")

        fig.savefig(savepath)
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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = dim_embed,
            nhead = nhead,
            dim_feedforward = dim_embed * expansion_factor,
            dropout = dropout,
            batch_first = True,
            activation = 'gelu',
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm = nn.LayerNorm(dim_embed),
        )

    def forward(self, x, mask):
        x = self.encoder(
            x,
            src_key_padding_mask = mask,
        )
        return x

class Decoder(nn.Module):
    def __init__(
        self,
        dim_embed: int,
        nhead: int = 8,
        num_layers: int = 6,
        expansion_factor: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

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
            num_layers=num_layers,
            norm = nn.LayerNorm(dim_embed),
        )

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
        x = self.decoder(
            x,
            memory,
            tgt_mask = self.generate_square_subsequent_mask(x.shape[1],x.device),
            tgt_key_padding_mask = mask,
            memory_key_padding_mask = memory_mask,
            tgt_is_causal = True,
        )
        return x


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
        slots: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_padding_mask: torch.Tensor | None = None,
    ):
        """
        Args:
            slots: (B, S, D)
            encoder_out: (B, N, D)
            encoder_padding_mask: (B, N)

        Returns:
            slots: (B, S, D)
        """

        output = slots

        for layer in self.layers:
            output = layer(
                output,
                encoder_out,
                encoder_padding_mask=encoder_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output

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
        encoder_out: torch.Tensor,
        encoder_padding_mask: torch.Tensor | None = None,
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
            query=slots,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_padding_mask,
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


class ClassifierMultiplicity(nn.Module):
    def __init__(self,dim_embed,max_seq_len,mode):
        super().__init__()
        assert mode in ['hard','soft']

        self.dim_embed = dim_embed
        self.max_seq_len = max_seq_len
        self.mode = mode

        self.layers = nn.Sequential(
                nn.Linear(self.dim_embed,64),
                nn.Dropout(0.1),
                nn.GELU(),
                nn.Linear(64,64),
                nn.Dropout(0.1),
                nn.GELU(),
                nn.Linear(64,32),
                nn.Dropout(0.1),
                nn.GELU(),
                nn.Linear(32,self.max_seq_len),
            )
        if self.mode == 'hard':
            self.loss_function = nn.CrossEntropyLoss(reduction='none')
        if self.mode == 'soft':
            self.loss_function = nn.BCELoss(reduction='none')

    def forward(self,x):
        return self.layers(x)

    def loss(self,y,mask):
        t = mask.sum(dim=1) - 1
        return self.loss_function(y,t)

    def sample(self,x):
        probs = torch.softmax(self(x), dim=-1)
        samples = torch.multinomial(probs, num_samples=1).squeeze(-1)
        idx = torch.arange(self.max_seq_len, device=samples.device)
        mask = idx.unsqueeze(0) <= samples.unsqueeze(1)
        return mask



class ClassifierHead(nn.Module):
    def __init__(self,dim_embed,dim_encoding,n_classes):
        super().__init__()
        self.dim_embed = dim_embed
        self.dim_encoding = dim_encoding
        self.n_classes = n_classes

        self.layers = nn.Sequential(
                nn.Linear(self.dim_embed,64),
                nn.Dropout(0.1),
                nn.GELU(),
                nn.Linear(64,64),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(64,32),
                nn.GELU(),
                nn.Dropout(0.1),
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
    ):
        super().__init__()

        self.dim_embed = 64
        self.feature_dict = feature_dict
        assert set(self.feature_dict.keys()) == set(classification+regression), f'Mismatch beteen features in inputs ({list(self.feature_dict.keys())}) and their split into classification ({classification}) and regression ({regression})'
        self.teacher_forcing = 1.0
        assert multiplicity in ['none','hard','soft']

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
                    nn.Linear(shape,32),
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
            num_layers = 6,
            expansion_factor = 4,
            dropout = 0.1,
        )
        self.decoder = Decoder(
            dim_embed = self.dim_embed,
            nhead = 4,
            num_layers = 8,
            expansion_factor = 4,
            dropout = 0.1,
        )
        self.start_token = nn.Parameter(
            torch.zeros(1,1,self.dim_embed),
            requires_grad = True,
        )

        # Multiplicity #
        if multiplicity == 'none':
            self.mult_encoding = None
            self.mult_head = None
        if multiplicity in ['hard','soft']:
            self.mult_token = nn.Parameter(
                torch.randn(1,1,self.dim_embed),
                requires_grad = True,
            )
            self.mult_encoding = CrossEncoder(
                dim_embed = self.dim_embed,
                nhead = 4,
                num_layers = 2,
                expansion_factor = 4,
                dropout = 0.1,
            )
            self.mult_head = ClassifierMultiplicity(
                dim_embed = self.dim_embed,
                max_seq_len = self.max_seq_len,
                mode = multiplicity,
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
                num_layers = 2,
                expansion_factor = 4,
                dropout = 0.1,
            )
            self.time_head = zuko.flows.NSF(
                features = 1,
                context = self.dim_embed,
                bins = 8,
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
            self.regression_heads[feature] = zuko.flows.NSF(
                features = len(self.feature_dict[feature]),
                context = self.dim_embed + dim_add_context,
                bins = 16,
                transforms = 5,
                randperm = True,
                passes = 2,
                hidden_features = [256,256],
            )
            dim_add_context += len(self.feature_dict[feature])

        # Classification heads #
        self.classification_heads = nn.ModuleDict()
        for feature in classification:
            assert feature in self.feature_dict.keys()
            self.classification_heads[feature] = ClassifierHead(
                dim_embed = self.dim_embed + dim_add_context,
                n_classes = len(self.feature_dict[feature]),
                dim_encoding = 16,
            )
            dim_add_context += self.classification_heads[feature].dim_encoding

        # Miscellanious #
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.01)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.surrogate_loss = []

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
        # Sample batch for teacher relaxing
        # (before preprocessing because done in sample as well)
        if self.teacher_forcing < 1:
            with torch.no_grad():
                samples,sample_mask,_ = self.sample(batch)
            # replace (avoiding mask)
            probs = torch.rand(sample_mask.shape).to(sample_mask.device)
            probs[~sample_mask] = 1
            probs[~batch['reco']['mask']] = 1
            replace_mask = probs < (1-self.teacher_forcing)
        # Preprocessing #
        batch = self.apply_preprocessing(batch)
        # Select (preprocessed) targets #
        reco_target = batch['reco']['data'].clone()
        mask_target = batch['reco']['mask'].clone()
        # Eventually replace some inputs #
        # (now that the targets are cloned)
        if self.teacher_forcing < 1:
            batch['reco']['data'][replace_mask,:] = samples[replace_mask,:]
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
        condition = F.dropout(condition, p=0.1, training=self.training)
        # Pass through regression head / flow #
        reg_losses = {}
        for feature in self.regression_heads.keys():
            t = torch.index_select(
                input = reco_target,
                dim = -1,
                index = self.feature_dict[feature].to(reco_target.device),
            )
            reg_losses[feature] = - (
                self.regression_heads[feature](condition).log_prob(t) * mask_target
            ).sum(dim=-1)
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
        # Multiplicity classifier #
        if self.mult_head is not None:
            mult_enc = self.mult_encoding(
                slots = self.mult_token.expand(x_enc.shape[0], -1, -1),
                encoder_out = x_enc,
                encoder_padding_mask = ~m_enc,
            ).squeeze(dim=1)
            y = self.mult_head(mult_enc)
            mult_losses = self.mult_head.loss(y,mask_target)
        else:
            mult_losses = None

        # Process time flow #
        if 'time' in batch.keys():
            time_enc = self.time_encoding(
                slots = self.time_token.expand(x_enc.shape[0], -1, -1),
                encoder_out = x_enc,
                encoder_padding_mask = ~m_enc,
            ).squeeze(dim=1)
            time_enc = F.dropout(time_enc, p=0.1, training=self.training)
            time_losses = - self.time_head(time_enc).log_prob(batch['time'])
        else:
            time_losses = None

        return class_losses, reg_losses, mult_losses, time_losses

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
        plotter: LossPlotting = None
    ):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if valid_dataset is None:
            valid_loader = None
        else:
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.to(self.device)

        assert teacher_forcing > 0 and teacher_forcing <= 1
        print(f"Surrogate Training: {n_epochs=}, {lr=}, {batch_size=}, {teacher_forcing=}")

        for epoch in range(n_epochs):
            # Training #
            self.train()
            self.teacher_forcing = teacher_forcing
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
            if self.mult_head is not None:
                train_losses['mult'] = torch.zeros(len(train_loader))
            if self.time_head is not None:
                train_losses['time'] = torch.zeros(len(train_loader))
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
                tot_loss = sum(tot_losses).mean()
                train_losses['total'][batch_idx] = tot_loss.item()
                # Optimizer step #
                self.optimizer.zero_grad()
                tot_loss.backward()
                self.optimizer.step()
            # Validation #
            self.eval()
            self.teacher_forcing = 1.0
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
                if self.mult_head is not None:
                    valid_losses['mult'] = torch.zeros(len(valid_loader))
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
                if self.mult_head is not None:
                    valid_losses['mult'] = torch.zeros(1)

            # Printout #
            s = f"Surrogate epoch = {epoch:3d} - Loss = {train_losses['total'].mean():+7.5f} [{valid_losses['total'].mean():+7.5f}] : "
            for name in train_losses.keys():
                if name == 'total':
                    continue
                if name in self.loss_factors.keys():
                    factor = self.loss_factors[name]
                else:
                    factor = 1.
                s += f"{name} = {factor.item(): .3f} x {train_losses[name].mean():+7.5f} [{valid_losses[name].mean():+7.5f}] + "
            print (s)
            self.surrogate_loss.append(valid_losses['total'].mean())

            # Plotter #
            if plotter is not None:
                plotter.add_lr_value(lr)
                for name,losses in train_losses.items():
                    plotter.add_train_value(name,losses.mean().item())
                for name,losses in valid_losses.items():
                    plotter.add_valid_value(name,losses.mean().item())

    def sample(self,batch):
        # Preprocessing #
        batch = self.apply_preprocessing(batch)
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
        samples = torch.zeros_like(batch['reco']['data']).to(self.device)
        if self.mult_head is None:
            samples_mask = batch['reco']['mask']
        else:
            mult_enc = self.mult_encoding(
                slots = self.mult_token.expand(x_enc.shape[0], -1, -1),
                encoder_out = x_enc,
                encoder_padding_mask = ~m_enc,
            ).squeeze(dim=1)
            samples_mask = self.mult_head.sample(mult_enc)
            m_dec = torch.cat(
                [
                    torch.full(
                        (samples_mask.shape[0],1),
                        fill_value = True,
                    ).to(batch['true']['mask'].device),
                    samples_mask,
                ],
                dim = 1,
            )
        # Sequential generation #
        for i in range(x_dec.shape[1]):
            # Pass through decoder, obtain new context #
            condition = self.decoder(
                x = x_dec[:,:i+1,:],
                mask = ~m_dec[:,:i+1],
                memory = x_enc,
                memory_mask = ~m_enc,
            )[:,i,:]
            particle = torch.zeros((samples.shape[0],samples.shape[2])).to(self.device)
            # Regression heads #
            for feature in self.regression_heads.keys():
                y = self.regression_heads[feature](condition).rsample()
                idx = self.feature_dict[feature]
                particle[...,idx] = y
                condition = torch.cat([condition,y],dim=-1)
            # Classification heads #
            for feature in self.classification_heads.keys():
                y = self.classification_heads[feature](condition)
                idx = self.feature_dict[feature]
                particle[...,idx] = y
                condition = torch.cat(
                    [
                        condition,
                        self.classification_heads[feature].encode(y)
                    ],
                    dim = -1,
                )
            # Add to samples and pass through embedding for new token #
            samples[:,i,:] = particle
            if i < x_dec.shape[1] - 1:
                x_dec[:,i+1,:] = self.embeddings['reco'](particle)
        # Undo preprocessing #
        samples = samples * self.stds['reco'] + self.means['reco']
        # Zero out missing particles #
        samples = (samples * samples_mask.unsqueeze(-1))

        return samples,samples_mask,times


    @torch.no_grad()
    def inference(
        self,
        dataset,
        batch_size,
        oversampling: int = 1,
    ):
        self.eval()
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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
                samples,mask,times = self.sample(batch)
                # Record #
                ia = batch_idx * batch_size
                ib = (batch_idx + 1) * batch_size

                all_samples[io,ia:ib] = samples.cpu()
                all_masks[io,ia:ib] = mask.cpu()
                if times is not None:
                    all_times[io,ia:ib] = times.cpu()

        return all_samples,all_masks,all_times


def validation_plot(
    loss_matching,
    savepath,
    true_part,
    true_mask,
    reco_part,
    reco_mask,
    samp_part,
    samp_mask,
    reco_time = None,
    samp_time = None,
):
    ncols = 2
    if samp_time is not None:
        ncols += 1
    fig,axs = plt.subplots(ncols=ncols,figsize=(6*ncols,5))
    plt.subplots_adjust(left=0.05,right=0.95,top=0.9,bottom=0.1,wspace=0.3)

    # Matching loss plot #
    reco_loss = loss_matching(
        true_part,
        true_mask,
        reco_part,
        reco_mask,
    ).repeat(samp_part.shape[0])
    sample_loss = torch.cat(
        [
            loss_matching(
                true_part,
                true_mask,
                samp_part[i],
                samp_mask[i],
            )
            for i in range(samp_part.shape[0])
        ]
    )

    min_loss = min(reco_loss.min(),sample_loss.min())
    max_loss = max(reco_loss.max(),sample_loss.max())
    bins = np.logspace(np.log10(min_loss),np.log10(max_loss),51)
    H = axs[0].hist2d(
        reco_loss.numpy(),
        sample_loss.numpy(),
        bins = bins,
        norm = matplotlib.colors.LogNorm(),
    )
    axs[0].plot(
        [min_loss,max_loss],
        [min_loss,max_loss],
        linestyle = '--',
        color = 'grey',
        linewidth = 2,
    )
    cbar = plt.colorbar(H[3],ax=axs[0])
    cbar.set_label('Count',fontsize=16)
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Reco matching loss',fontsize=16)
    axs[0].set_ylabel('Surrogate matching loss',fontsize=16)

    # Confusion matrix for multiplicity #
    N_reco = reco_mask.sum(dim=-1).repeat(samp_mask.shape[0])
    N_samp = samp_mask.sum(dim=-1).ravel()
    cm = confusion_matrix(N_samp, N_reco, labels=np.arange(reco_mask.shape[-1]))

    disp = ConfusionMatrixDisplay(
        confusion_matrix = cm,
        display_labels = np.arange(reco_mask.shape[-1]),
    )
    disp.plot(
        ax = axs[1],
        cmap = 'Blues',
        colorbar = True,
        values_format = 'd',
        text_kw={'fontsize': 8},
    )
    axs[1].invert_yaxis()
    axs[1].set_ylabel('Reco number of particles',fontsize=16)
    axs[1].set_xlabel('Surrogate number of particles',fontsize=16)

    # Time plot if done #
    if samp_time is not None:
        samp_times = samp_time.squeeze(-1).permute(1,0).clone()
        true_times = reco_time.clone()
        N = math.lcm(samp_times.shape[-1],true_times.shape[-1])
        samp_times = samp_times.repeat_interleave(N // samp_times.shape[-1], dim=1)
        true_times = true_times.repeat_interleave(N // true_times.shape[-1], dim=1)
        bins = np.linspace(
            min(samp_times.min(),true_times.min()),
            max(samp_times.max(),true_times.max()),
            51
        )
        H = axs[2].hist2d(
            true_times.ravel().numpy(),
            samp_times.ravel().numpy(),
            bins = bins,
            norm = matplotlib.colors.LogNorm(),
        )
        cbar = plt.colorbar(H[3],ax=axs[2])
        cbar.set_label('Count',fontsize=16)
        axs[2].set_xlabel('Reco inference time',fontsize=16)
        axs[2].set_ylabel('Sampled inference time',fontsize=16)


    fig.savefig(savepath)
    print (f'Validation plot saved as {savepath}')


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

