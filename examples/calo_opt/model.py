import os
import sys
import math
import pathlib
from copy import deepcopy
import numpy as np
from tqdm.auto import tqdm
import torch
from torch import nn
import torch_geometric
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix

from fastgraphcompute.gnn_ops import GravNetOp
from fastgraphcompute.object_condensation import ObjectCondensation

sys.path.append(os.path.abspath(pathlib.Path(__file__).parent.parent))
from dataset import CaloGraphDataset

class GravBlock(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_embed: int,
        dim_out: int,
        dim_space: int,
        dim_propagate: int,
        k: int,
        graphnorm: bool = False,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim_in,dim_embed),
            nn.ELU(),
            nn.Linear(dim_embed,dim_embed),
            nn.ELU(),
            nn.Linear(dim_embed,dim_embed),
        )
        self.grav_layer = GravNetOp(
            in_channels = dim_embed,
            out_channels = dim_out,
            space_dimensions = dim_space,
            propagate_dimensions = dim_propagate,
            k = k,
        )
        self.graphnorm = torch_geometric.nn.GraphNorm(dim_out) if graphnorm else None

    def forward(self,x,row_splits):
        x = self.mlp(x)
        x, _,_,_ = self.grav_layer(x,row_splits)
        if self.graphnorm is not None:
            x = self.graphnorm(x)
        return x

class BufferValue(nn.Module):
    def __init__(self,values):
        super().__init__()
        if torch.is_tensor(values):
            self.values = nn.Parameter(values, requires_grad=False)
        elif isinstance(values, (int, float, list, tuple)):
            self.values = nn.Parameter(torch.tensor([values]), requires_grad=False)
        else:
            raise TypeError(f'Type {type(values)} not implemented')


class BufferModule(nn.Module):
    def __init__(self,dict_values):
        super().__init__()

        self.data = nn.ModuleDict()
        for key,values in dict_values.items():
            if isinstance(values,dict):
                self.data[key] = BufferModule(values)
            else:
                self.data[key] = BufferValue(values)

    def __getitem__(self,key):
        return self.data[key]

    def keys(self):
        return self.data.keys()

class GravNetModel(nn.Module):
    def __init__(
        self,
        inputs,
        regression,
        classification,
        loss_factors,
        shapes,
        means,
        stds,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        self.shapes = BufferModule(shapes)
        self.loss_factors = BufferModule(loss_factors)
        self.means = BufferModule(means)
        self.stds = BufferModule(stds)

        self.dim_embed = 64
        self.dim_out = 32
        self.dim_space = 3
        self.dim_propagate = 16
        self.n_blocks = 4
        self.k = 8
        self.graphnorm = True

        self.init_layers = nn.ModuleDict(
            {
                name: nn.Linear(self.shapes['hits'][name].values[0],16)
                for name in inputs
            }
        )
        if 'global_params' in self.shapes.keys():
            self.init_layers['global_params'] = nn.Linear(self.shapes['global_params'].values[0],16)
        self.dim_in = 16 * len(self.init_layers)
        self.init_dense = nn.Sequential(
            nn.BatchNorm1d(self.dim_in),
            nn.Linear(self.dim_in, self.dim_embed),
            nn.BatchNorm1d(self.dim_embed),
            nn.ELU(),
            nn.Linear(self.dim_embed, self.dim_embed),
            nn.BatchNorm1d(self.dim_embed),
            nn.ELU(),
            nn.Linear(self.dim_embed, self.dim_out),
        )

        self.blocks = nn.ModuleList(
            [
                GravBlock(
                    dim_in = self.dim_out,
                    dim_embed = self.dim_embed,
                    dim_out = self.dim_out,
                    dim_space = self.dim_space,
                    dim_propagate = self.dim_propagate,
                    k = self.k,
                    graphnorm = self.graphnorm,
                )
                for i in range(self.n_blocks)
            ]
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.n_blocks * self.dim_out, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
        )

        self.heads = nn.ModuleDict(
            {
                'beta' : nn.Sequential(
                    nn.Linear(256,128),
                    nn.BatchNorm1d(128),
                    nn.ELU(),
                    nn.Linear(128,64),
                    nn.BatchNorm1d(64),
                    nn.ELU(),
                    nn.Linear(64,1),
                    nn.Sigmoid(),
                ),
                'embedding' : nn.Sequential(
                    nn.Linear(256,128),
                    nn.BatchNorm1d(128),
                    nn.ELU(),
                    nn.Linear(128,64),
                    nn.BatchNorm1d(64),
                    nn.ELU(),
                    nn.Linear(64,3),
                ),
            }
        )
        self.loss_functions = nn.ModuleDict(
            {
                'beta' : ObjectCondensation(q_min=0.1, s_B=1.0),
            }
        )
        for name in regression:
            self.heads[name] = nn.Sequential(
                    nn.Linear(256,128),
                    nn.BatchNorm1d(128),
                    nn.ELU(),
                    nn.Linear(128,64),
                    nn.BatchNorm1d(64),
                    nn.ELU(),
                    nn.Linear(64,self.shapes['particles'][name].values[0]),
                )
            nn.Linear(256,self.shapes['particles'][name].values[0])
            self.loss_functions[name] = nn.MSELoss(reduction='none')
        for name in classification:
            self.heads[name] = nn.Sequential(
                    nn.Linear(256,128),
                    nn.BatchNorm1d(128),
                    nn.ELU(),
                    nn.Linear(128,64),
                    nn.BatchNorm1d(64),
                    nn.ELU(),
                    nn.Linear(64,self.shapes['particles'][name].values[0]),
                )
            self.loss_functions[name] = nn.CrossEntropyLoss(reduction='none') if self.shapes['particles'][name].values[0] > 1 else nn.BCELoss(reduction='mean')

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.device = torch.device(device)

    def set_to_device(self,device):
        self.device = device
        self.to(self.device)

    def forward(self, batch):
        # Init and concat #
        xs = [
            self.init_layers[key](batch['hits'][key])
            for key in self.init_layers.keys() if key != 'global_params'
        ]
        x = torch.cat(xs,dim=-1)
        if 'global_params' in batch.keys():
            x = torch.cat(
                [
                    x,
                    self.init_layers['global_params'](
                        batch['global_params'][batch['hits']['batch']],
                    )
                ],
                dim = -1
            )
        x = self.init_dense(x)
        # Apply blocks and keep track of outputs #
        ys = []
        for block in self.blocks:
            x = block(x,row_splits=batch['hits']['ptr'])
            ys.append(x)
        # Concat and pass through last layer and heads#
        y = torch.cat(ys,dim=-1)
        y = self.mlp(y)
        return {
            key : head(y)
            for key,head in self.heads.items()
        }

    def loss(self,batch,outputs):
        # Get condensation loss #
        beta = outputs['beta']
        embedding = outputs['embedding']

        L_att, L_rep, L_beta, _, _ = self.loss_functions['beta'](
            beta = beta,
            coords = embedding,
            asso_idx = batch['hits']['labels'].to(torch.int64),
            row_splits = batch['hits']['ptr'],
        )

        losses = {
            'attraction' : L_att,
            'repulsion' : L_rep,
            'beta' : L_beta,
        }

        edge_index = batch[('hits','link','particles')].edge_index
        for key in self.loss_functions.keys():
            if key == 'beta':
                continue
            pred = outputs[key]
            true = batch['particles'][key][edge_index[1]]
            loss = self.loss_functions[key](pred,true)
            if loss.dim() > 1:
                loss = loss.sum(dim=-1)
            losses[key] = (beta.ravel() * loss).mean()

        return losses

    def apply_preprocessing(self,batch):
        for node in batch.node_types:
            if node not in self.means.keys() or node not in self.stds.keys():
                continue
            for key,values in batch[node].items():
                if key in self.means[node].keys() and key in self.stds[node].keys():
                    batch[node][key] = (values - self.means[node][key].values) / self.stds[node][key].values
        if 'global_params' in batch.keys() and 'global_params' in self.means.keys() and 'global_params' in self.stds.keys():
            batch['global_params'] = (batch['global_params'] - self.means['global_params'].values) / self.stds['global_params'].values
        return batch

    @staticmethod
    def print_batch_size(batch):
        total_bytes = 0
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                total_bytes += value.numel() * value.element_size()
            # handle nested HeteroDataBatch
            elif hasattr(value, "__dict__"):
                total_bytes += sum(v.numel() * v.element_size()
                                   for v in value.__dict__.values()
                                   if isinstance(v, torch.Tensor))
        print(f"Batch size: {total_bytes / (1024**2):.2f} MB")

    def train_model(
        self,
        train_dataset,
        valid_dataset,
        batch_size: int,
        n_epochs: int,
        lr: float,
        n_batches: int = math.inf,
        annealing = None,
        plotter = None,
    ):
        print(f"Reconstruction Training: {n_epochs=} {lr=}, {batch_size=}")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        n_batches_train = min(n_batches,len(train_loader))
        n_batches_valid = min(n_batches,len(valid_loader))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.to(self.device)
        self.train()

        if 'cuda' in self.device:
            batch = next(iter(train_loader))
            batch = batch.to(self.device)
            with torch.amp.autocast(self.device):
                out = self(batch)
            print("Peak memory during forward:", torch.cuda.max_memory_allocated() / 1e9, "GB")

        def get_factor(name,epoch):
            if name in self.loss_factors.keys():
                factor = self.loss_factors[name].values.item()
            else:
                factor = 1.
            if annealing is not None and name not in ['total','attraction','repulsion','beta']:
                factor *= annealing()
            return factor

        for epoch in range(n_epochs):
            # Training #
            self.train()
            train_losses = {
                'total' : torch.zeros(n_batches_train),
                'attraction' : torch.zeros(n_batches_train),
                'repulsion' : torch.zeros(n_batches_train),
                'beta' : torch.zeros(n_batches_train),
                **{key: torch.zeros(n_batches_train) for key in self.loss_functions.keys() if key != 'beta'}
            }
            for batch_idx, batch in tqdm(enumerate(train_loader),total=n_batches_train,desc='Training batches',leave=False):
                if batch_idx >= n_batches_train:
                    break
                # Move to device #
                batch = batch.to(self.device)
                # Preprocessing # # TODO : move to forward
                batch = self.apply_preprocessing(batch)
                # Process #
                outputs = self(batch)
                # Losses #
                losses = self.loss(batch,outputs)
                # Total loss #
                loss_values = []
                for key,loss in losses.items():
                    loss_values.append(get_factor(key,epoch) * loss)
                loss_tot = sum(loss_values)
                self.optimizer.zero_grad()
                loss_tot.backward()
                self.optimizer.step()
                # Record #
                for key,val in losses.items():
                    train_losses[key][batch_idx] = val.item()
                train_losses['total'][batch_idx] = loss_tot.item()
            # Validation #
            self.eval()
            valid_losses = {
                'total' : torch.zeros(n_batches_valid),
                'attraction' : torch.zeros(n_batches_valid),
                'repulsion' : torch.zeros(n_batches_valid),
                'beta' : torch.zeros(n_batches_valid),
                **{key: torch.zeros(n_batches_valid) for key in self.loss_functions.keys() if key != 'beta'}
            }
            for batch_idx, batch in tqdm(enumerate(valid_loader),total=n_batches_valid,desc='Validation batches',leave=False):
                if batch_idx >= n_batches_valid:
                    break
                # Move to device #
                batch = batch.to(self.device)
                # Preprocessing # TODO : move to forward
                batch = self.apply_preprocessing(batch)
                # Process #
                with torch.no_grad():
                    outputs = self(batch)
                # Losses #
                losses = self.loss(batch,outputs)
                # Total loss #
                loss_values = []
                for key,loss in losses.items():
                    loss_values.append(get_factor(key,epoch) * loss)
                loss_tot = sum(loss_values)
                # Record #
                for key,val in losses.items():
                    valid_losses[key][batch_idx] = val.item()
                valid_losses['total'][batch_idx] = loss_tot.item()
            # Print losses #
            s = f"Reco epoch = {epoch:3d} - Loss = {train_losses['total'].mean():7.5f} [{valid_losses['total'].mean():7.5f}] : "
            for key in train_losses.keys():
                if key == 'total':
                    continue
                factor = get_factor(key,epoch)
                s += f"{key} = {factor:.3f} x {train_losses[key].mean():7.5f} [{valid_losses[key].mean():7.5f}] + "
            print (s)

            if plotter is not None:
                for key in train_losses.keys():
                    plotter.add_train_value(f'{key}',train_losses[key].mean())
                    plotter.add_valid_value(f'{key}',valid_losses[key].mean())
                plotter.add_lr_value('lr',lr)
            if annealing is not None:
                annealing.new_epoch()

    @torch.no_grad()
    def inference(
        self,
        dataset,
        batch_size = 64,
        t_beta = 0.1,
        t_dist = 0.8,
    ):
        """ Apply the model in batches this is necessary because the model is too large to apply it to the
        whole dataset at once. The model is applied to the dataset in batches and the results are concatenated
        (the batch size is a hyperparameter).
        """
        print (f'Condensation inference with t_beta = {t_beta} and t_dist = {t_dist}')

        self.to(self.device)
        self.eval()

        # Perform inferences #
        loader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
        data_list = []
        for batch_idx, batch in tqdm(enumerate(loader),total=len(loader),desc='Batches',leave=True):
            # Move to device #
            proc_batch = deepcopy(batch).to(self.device)
            # Preprocessing #
            proc_batch = self.apply_preprocessing(proc_batch)
            # Process #
            outputs = self(proc_batch)
            # Add output to batch (hack with slice_dict) #
            batch._slice_dict['predictions'] = {}
            batch._inc_dict['predictions'] = {}
            for key,val in outputs.items():
                setattr(batch['predictions'],key,val.cpu())
                batch._slice_dict['predictions'][key] = batch['hits'].ptr
                batch._inc_dict['predictions'][key] = torch.zeros(len(batch))
            batch['predictions'].num_nodes = batch['hits'].num_nodes
            batch['predictions'].batch = batch['hits'].batch
            batch['predictions'].ptr = batch['hits'].ptr
            # Batch->data_list #
            data_list.extend(batch.to_data_list())

        def condense(beta,embedding,t_beta,t_dist):
            beta_order = beta.argsort(descending=True,dim=0)
            vertex_indices = []
            vertex_embeddings = torch.tensor([])
            for idx in beta_order:
                if beta[idx] < t_beta:
                    break
                if len(vertex_indices) == 0:
                    vertex_indices.append(idx)
                    vertex_embeddings = embedding[idx]
                else:
                    prev_x = torch.concatenate(
                        [
                            embedding[v_idx] for v_idx in vertex_indices
                        ],
                        dim = 0,
                    )
                    vertex_emb = embedding[idx]
                    dist = torch.norm(vertex_emb- vertex_embeddings, dim=1)
                    if dist.min() > t_dist:
                        vertex_indices.append(idx)
                        vertex_embeddings = torch.cat([vertex_embeddings,vertex_emb],dim=0)
            vertex_indices = torch.tensor(vertex_indices).to(torch.int)
            return vertex_indices,vertex_embeddings

        if t_dist == 'auto':
            t_dists = torch.linspace(0,1,51)
            indices = torch.randperm(len(data_list))[:200]
            results = torch.zeros(len(t_dists))
            for i,t in tqdm(enumerate(t_dists),desc='Optimising t_dist',total=len(t_dists),leave=True):
                n_true = []
                n_reco = []
                for idx in indices:
                    n_true.append(data_list[idx]['particles'].num_nodes)
                    vertex_indices,_ = condense(
                        beta = data_list[idx]['predictions']['beta'],
                        embedding = data_list[idx]['predictions']['embedding'],
                        t_beta = t_beta,
                        t_dist = t,
                    )
                    n_reco.append(len(vertex_indices))
                cm = torch.from_numpy(confusion_matrix(n_true,n_reco))
                results[i] = torch.diagonal(cm).sum() / cm.sum()
            t_dist = round(t_dists[results.argmax()].item(),5)
            print (f'Found optimal t_dist = {t_dist} : fraction of diagonal elements = {results.max().item()*100:5.2f}%')


        for data in tqdm(data_list,desc='Events',leave=True):
            # Condensation #
            vertex_indices,vertex_embeddings = condense(
                beta = data['predictions']['beta'],
                embedding = data['predictions']['embedding'],
                t_beta = t_beta,
                t_dist = t_dist,
            )
            data['vertices'].idx = vertex_indices
            data['vertices'].embedding = vertex_embeddings
            # Obtain particle prediction for the condensation vertices #
            for key in self.heads.keys():
                if key in ['beta','embedding']:
                    continue
                out = data['predictions'][key]
                out = out[vertex_indices]
                out = out * self.stds['particles'][key].values.cpu() + self.means['particles'][key].values.cpu()
                setattr(data['vertices'],key,out)

        return CaloGraphDataset.from_data_list(data_list), t_dist


