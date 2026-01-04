from copy import deepcopy
import numpy as np
from tqdm.auto import tqdm
import torch
from torch import nn
from torch_geometric.loader import DataLoader

from .dataset import CaloGraphDataset
from fastgraphcompute.gnn_ops import GravNetOp
from fastgraphcompute.object_condensation import ObjectCondensation

class GravBlock(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_embed: int,
        dim_out: int,
        dim_space: int,
        dim_propagate: int,
        k: int,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim_in,dim_embed),
            nn.ReLU(),
            nn.Linear(dim_embed,dim_embed),
            nn.ReLU(),
            nn.Linear(dim_embed,dim_embed),
        )
        self.grav_layer = GravNetOp(
            in_channels = dim_embed,
            out_channels = dim_out,
            space_dimensions = dim_space,
            propagate_dimensions = dim_propagate,
            k = k,
        )

    def forward(self,x,row_splits):
        x = self.mlp(x)
        x, _,_,_ = self.grav_layer(x,row_splits)
        return x

#           import torch
#   from fastgraphcompute.gnn_ops import GravNetOp
#
#   model = GravNetOp(in_channels=8, out_channels=16, space_dimensions=4, propagate_dimensions=8, k=20)
#   input_tensor = torch.rand(32, 8)
#   # row split format, cutting the 32 x 8 array into individual samples / events
#   # one with 18 entries, one with 14 entries.
#   row_splits = torch.tensor([0, 18, 32], dtype=torch.int32)
#   output, neighbor_idx, distsq, S_space = model(input_tensor, row_splits)
#   print(output.shape)  # Expected output: (32, 16)

class GravNetModel(nn.Module):
    def __init__(
        self,
        inputs,
        reg_outputs,
        cls_outputs,
        loss_factors,
        shapes,
        means,
        stds,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        self.shapes = shapes
        self.save_dict("loss_factors",loss_factors)
        self.save_dict("means",means)
        self.save_dict("stds",stds)

        self.dim_embed = 64
        self.dim_out = 32
        self.dim_space = 4
        self.dim_propagate = 16
        self.dim_in = 16 * len(inputs)
        self.n_blocks = 4
        self.k = 16

        self.init_layers = nn.ModuleDict(
            {
                f'__{inp}__': nn.Linear(self.shapes[inp],16)
                for inp in inputs
            }
        )

        self.blocks = nn.ModuleList(
            [
                GravBlock(
                    dim_in = self.dim_in if i == 0 else self.dim_out,
                    dim_embed = self.dim_embed,
                    dim_out = self.dim_out,
                    dim_space = self.dim_space,
                    dim_propagate = self.dim_propagate,
                    k = self.k,
                )
                for i in range(self.n_blocks)
            ]
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.n_blocks * self.dim_out, 128),
            nn.ReLU(),
            nn.Linear(self.n_blocks * self.dim_out, 128),
            nn.ReLU(),
        )

        self.heads = nn.ModuleDict(
            {
                'beta' : nn.Sequential
                (
                    nn.Linear(128,1),
                    nn.Sigmoid(),
                ),
                'embedding' : nn.Sequential
                (
                    nn.Linear(128,3),
                ),
            }
        )
        self.loss_functions = nn.ModuleDict(
            {
                'beta' : ObjectCondensation(q_min=0.1, s_B=1.0),
            }
        )
        for out in reg_outputs:
            self.heads[out] = nn.Linear(128,self.shapes[out])
            self.loss_functions[out] = nn.MSELoss(reduction='none')
        for out in cls_outputs:
            self.heads[out] = nn.Linear(128,self.shapes[out])
            self.loss_functions[out] = nn.CrossEntropyLoss(reduction='none') if self.shapes[out] > 1 else nn.BCELoss(reduction='mean')

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        self.device = torch.device(device)

    def forward(self, batch):
        # Init and concat #
        x = torch.cat(
            [
                self.init_layers[key](batch[key.replace('__','')])
                for key in self.init_layers.keys()
            ],
            dim = -1,
        )
        # Apply blocks and keep track of outputs #
        ys = []
        for block in self.blocks:
            x = block(x,row_splits=batch['ptr'])
            ys.append(x)
        # Concat and pass through last layer and heads#
        y = torch.cat(ys,dim=-1)
        y = self.mlp(y)
        return {
            key : head(y)
            for key,head in self.heads.items()
        }

    def save_dict(self,name,values_dict):
        setattr(self,name,nn.ModuleDict())
        buffer = getattr(self,name)
        for key,val in values_dict.items():
            if not torch.is_tensor(val):
                val = torch.tensor(val,dtype=torch.float)
            mod = nn.Module()
            mod.register_buffer("value", val)
            buffer[f'__{key}__'] = mod

    def loss(self,batch,outputs):
        # Get condensation loss #
        beta = outputs['beta']
        embedding = outputs['embedding']

        L_att, L_rep, L_beta, _, _ = self.loss_functions['beta'](
            beta = beta,
            coords = embedding,
            asso_idx = batch['labels'].to(torch.int64),
            row_splits = batch['ptr'],
        )

        losses = {
            'attraction' : L_att,
            'repulsion' : L_rep,
            'beta' : L_beta,
        }

        for key in self.loss_functions.keys():
            if key == 'beta':
                continue
            loss = self.loss_functions[key](
                outputs[key],
                batch[key][batch['batch']],
            )
            if loss.dim() > 1:
                loss = loss.mean(dim=-1)
            losses[key] = (beta.ravel() * loss).mean()

        return losses


    def train_model(
        self,
        train_dataset,
        valid_dataset,
        batch_size: int,
        n_epochs: int,
        lr: float,
    ):
        print(f"Reconstruction Training: {lr=}, {batch_size=}")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.to(self.device)
        self.train()

        for epoch in range(n_epochs):
            # Training #
            self.train()
            train_losses = {
                'total' : torch.zeros(len(train_loader)),
                'attraction' : torch.zeros(len(train_loader)),
                'repulsion' : torch.zeros(len(train_loader)),
                'beta' : torch.zeros(len(train_loader)),
                **{key: torch.zeros(len(train_loader)) for key in self.loss_functions.keys() if key != 'beta'}
            }
            for batch_idx, batch in tqdm(enumerate(train_loader),total=len(train_loader),desc='Training batches',leave=False):
                # Move to device #
                batch = {key:val.to(self.device) for key,val in batch.items()}
                # Preprocessing #
                batch = {
                    key : (val - self.means[f'__{key}__'].value) / self.stds[f'__{key}__'].value
                        if f'__{key}__' in self.means.keys() else val
                    for key,val in batch.items()
                }
                # Process #
                outputs = self(batch)
                # Losses #
                losses = self.loss(batch,outputs)
                # Total loss #
                loss_values = []
                for key,loss in losses.items():
                    if key in self.loss_factors.keys():
                        factor = self.loss_factors[f'__{key}__'].value
                    else:
                        factor = 1.
                    loss_values.append(factor * loss)
                loss_tot = sum(loss_values)
                self.optimizer.zero_grad()
                loss_tot.backward()
                self.optimizer.step()
                # Record #
                for key,val in losses.items():
                    train_losses[key][batch_idx] = val.item()
                train_losses['total'][batch_idx] = loss_tot.item()

            s = f"Reco epoch = {epoch:4d} - Loss = {train_losses['total'].mean():7.5f} : "
            for key,loss in train_losses.items():
                if key == 'total':
                    continue
                if f'__{key}__' in self.loss_factors.keys():
                    factor = self.loss_factors[f'__{key}__'].value
                else:
                    factor = 1.
                s += f"{key} = {factor:.1f} x {loss.mean():7.5f} [] + "
            print (s)
#            # Validation #
#            self.eval()
#            valid_losses = torch.zeros(len(valid_loader))
#            for batch_idx, (detector_parameters, x, c, y) in enumerate(valid_loader):
#                detector_parameters: torch.Tensor = detector_parameters.to(self.device)
#                x: torch.Tensor = x.to(self.device)
#                c: torch.Tensor = c.to(self.device)
#                y: torch.Tensor = y.to(self.device)
#                y_pred: torch.Tensor = self(detector_parameters, x, c)
#                loss_per_event = self.loss(
#                    train_dataset.unnormalize_target(y),
#                    train_dataset.unnormalize_target(y_pred)
#                )
#                loss = loss_per_event.clone().mean()
#                valid_losses[batch_idx] = loss.item()
#
#            print(f"Reco Epoch: {epoch:4d} - Loss: {train_losses.mean():8.3f} - Val loss {valid_losses.mean():8.3f}")
#

    def inference(
        self,
        dataset,
        t_beta = 0.1,
        t_dist = 0.8,
    ):
        """ Apply the model in batches this is necessary because the model is too large to apply it to the
        whole dataset at once. The model is applied to the dataset in batches and the results are concatenated
        (the batch size is a hyperparameter).
        """
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        num_nodes = [[dataset[i].num_nodes for i in range(len(dataset))]]
        results = {
            'beta' : torch.zeros((len(dataset),1)),
            'embedding' : torch.zeros((len(dataset),3)),
            **{
                key : torch.zeros((len(dataset),self.shapes[key]))
                for key in self.heads.keys() if key not in ['beta','embedding']
            }
        }

        self.to(self.device)
        self.eval()

        data_list = []
        for data in tqdm(dataset,desc='Events',leave=False):
            # Make single entry batch #
            batch = deepcopy(data)
            batch['batch'] = torch.zeros(data.num_nodes, dtype=torch.long)
            batch['ptr'] = torch.tensor([0, data.num_nodes], dtype=torch.long)
            # Move to device #
            batch = {key:val.to(self.device) for key,val in batch.items()}
            # Preprocessing #
            batch = {
                key : (val - self.means[f'__{key}__'].value) / self.stds[f'__{key}__'].value
                    if f'__{key}__' in self.means.keys() else val
                for key,val in batch.items()
            }
            # Process #
            outputs = self(batch)
            beta = outputs['beta'].cpu()
            embedding = outputs['embedding'].cpu()
            data['beta'] = beta
            # Inference #
            beta_order = beta.argsort(descending=True,dim=0)
            vertex_indices = []
            for idx in beta_order:
                if beta[idx] < t_beta:
                    break
                if len(vertex_indices) == 0:
                    vertex_indices.append(idx)
                else:
                    prev_x = torch.concatenate(
                        [
                            embedding[v_idx] for v_idx in vertex_indices
                        ],
                        dim = 0,
                    )
                    vertex_x = embedding[idx]
                    dist = torch.norm(prev_x - vertex_x, dim=1)
                    if dist.min() > t_dist:
                        vertex_indices.append(idx)
            vertex_indices = torch.tensor(vertex_indices)
            data['vertex_idx'] = vertex_indices
            # Obtain particle prediction for the condensation vertices #
            for key,out in outputs.items():
                if 'particle' in key:
                    out = out * self.stds[f'__{key}__'].value + self.means[f'__{key}__'].value
                    data[key.replace('particle','vertex')] = out[vertex_indices]

            # Save data #
            data_list.append(data)

        return CaloGraphDataset.from_data_list(data_list)



#        # Obtain predictions #
#        for batch_idx, batch in tqdm(enumerate(data_loader),total=len(data_loader),desc='Inference batches',leave=False):
#            # Move to device #
#            batch = {key:val.to(self.device) for key,val in batch.items()}
#            # Preprocessing #
#            batch = {
#                key : (val - self.means[f'__{key}__'].value) / self.stds[f'__{key}__'].value
#                    if f'__{key}__' in self.means.keys() else val
#                for key,val in batch.items()
#            }
#            # Process #
#            outputs = self(batch)
#            from IPython import embed; embed()
#            for key in outputs.keys():
#                results[key][batch_idx * batch_size: (batch_idx + 1) * batch_size] = outputs[key]
#
#        # Perform object condensation #
#        data_list = [dataset[i] for i in range(len(dataset))]
#
#        beta = results['beta']
#        embedding = results['embedding']
#
#        beta_order = beta.argsort(descending=True)
#        vertex_indices = []
#        for idx in beta_order:
#            if beta[idx] < t_beta:
#                break
#            from IPython import embed; embed()





