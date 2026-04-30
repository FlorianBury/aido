import json
import math
import os
import sys
import random
from tqdm import tqdm
import numpy as np
import torch

from torch_geometric.data import Data, HeteroData

import numpy as np
import pandas as pd
from G4Calo import GeometryDescriptor, run_batch
from minipandas import MiniFrame,concat

from dataset import CaloGraphDataset

class Simulation():
    def __init__(
            self,
            config: dict,
            parameter_dict: dict
        ):
        self.config = config
        self.parameter_dict = parameter_dict

        self.n_parts = self.config['number_simulations_per_particle']
        self.n_events = self.config['number_stitched_events']
        self.mean_poisson = self.config['mean_number_particles_per_event']
        self.min_dist = self.config['min_distance_cut']
        self.minEnergy_GeV = self.config['minEnergy_GeV']
        self.maxEnergy_GeV = self.config['maxEnergy_GeV']
        self.particle_types = self.config['particle_types']

        self.cw = GeometryDescriptor()

        for i in range(parameter_dict['num_layers']['current_value']):
            self.cw.addLayer(
                max(parameter_dict[f"thickness_absorber:{i}"]["current_value"], 1e-3),
                parameter_dict[f"material_absorber:{i}"]["current_value"],
                False,
                1,
            )
            self.cw.addLayer(
                max(parameter_dict[f"thickness_scintillator:{i}"]["current_value"], 1e-3),
                parameter_dict[f"material_scintillator:{i}"]["current_value"],
                True,
                parameter_dict[f"granularity_scintillator:{i}"]["current_value"],
            )

    def run_simulation(self):
        mfs = []
        for part_type in self.particle_types:
            print (f'Running {part_type}')
            mf: MiniFrame = run_batch(
                gd = self.cw,
                nEvents = self.n_parts,
                particleSpec = part_type,
                minEnergy_GeV = self.minEnergy_GeV,
                maxEnergy_GeV = self.maxEnergy_GeV,
                no_mp = True,
                manual_seed = self.parameter_dict["metadata"]["rng_seed"],
            )
            mfs.append(mf)
        mf = concat(mfs)
        # Type onehot encoding #
        unique_vals, indices = np.unique(mf['particle_type'], return_inverse=True)
        onehot = np.eye(len(unique_vals))[indices.ravel()]
        mf._data['particle_type'] = onehot
        return mf

    def stitch_one_event(self,mfs):
        # sensor positions should be the same for same set of simulated parameters
        x = torch.from_numpy(mfs[0]['sensor_x']) / 10
        y = torch.from_numpy(mfs[0]['sensor_y']) / 10
        z = torch.from_numpy(mfs[0]['sensor_z']) / 10
        dx = torch.from_numpy(mfs[0]['sensor_dx']) / 10
        dy = torch.from_numpy(mfs[0]['sensor_dy']) / 10
        dz = torch.from_numpy(mfs[0]['sensor_dz']) / 10
        layer = torch.from_numpy(mfs[0]['sensor_layer'])

        # Concatenate particle content #
        part_x = torch.concatenate(
            [
                torch.from_numpy(mf['particle_x']).squeeze(-1)
                for mf in mfs
            ],
            axis = 0,
        )
        part_y = torch.concatenate(
            [
                torch.from_numpy(mf['particle_y']).squeeze(-1)
                for mf in mfs
            ],
            axis = 0,
        )
        part_E = torch.concatenate(
            [
                torch.from_numpy(mf['particle_energy']).squeeze(-1)
                for mf in mfs
            ],
            axis = -1,
        )
        part_type = torch.concatenate(
            [
                torch.from_numpy(mf['particle_type'])
                for mf in mfs
            ],
            axis = 0,
        )

        # Stitch energies and extract particle index #
        Es = np.array([mf['sensor_energy'] for mf in mfs]) / 1000 # MeV -> GeV
        Es = torch.from_numpy(Es)
        E = Es.sum(dim=0)
        fracs = Es / (E + 1e-10)
        labels = torch.argmax(fracs, dim=0)

        # Set noise labels (-1) to hits with <5% true energy #
        #labels[fracs.max(dim=0).values < 0.05] = -1

        # Make mask or non-zero cells and save #
        mask = E > 0
        event = HeteroData()

        # Hit/calo info #
        event['hits'].pos = torch.stack(
            [x[mask],y[mask],z[mask]],
            dim = -1,
        ).to(torch.float32)
        event['hits'].E = E[mask].unsqueeze(-1).to(torch.float32)
        event['hits'].layer = layer[mask].unsqueeze(-1).to(torch.float32)
        event['hits'].cell = torch.stack(
            [dx[mask],dy[mask],dz[mask]],
            dim = -1,
        ).to(torch.float32)
        event['hits'].labels = labels[mask].unsqueeze(-1).to(torch.int64)
        event['hits'].num_nodes = event['hits'].pos.size(0)

        # Particle info #
        event['particles'].pos = torch.stack(
            [part_x,part_y],
            dim = -1,
        ).to(torch.float32)
        event['particles'].E = part_E.unsqueeze(-1).to(torch.float32)
        event['particles'].id = part_type.to(torch.float32)
        event['particles'].num_nodes = event['particles'].E.size(0)

        # Hit <-> particle link #
        #valid = labels >= 0  # ignore noise if needed
        event['hits', 'link', 'particles'].edge_index = torch.stack(
            [
                torch.arange(labels[mask].size(0)),
                labels[mask],
            ],
            dim=0,
        )

        return event

    def get_distances(self,mfs):
        pos = np.concatenate(
            [
                np.concatenate(
                    [
                        mf['particle_x'],
                        mf['particle_y'],
                    ],
                    axis = 1,
                )
                for mf in mfs
            ],
            axis = 0,
        )
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        dist = dist[np.triu_indices(dist.shape[0], k=1)]
        return dist

    def stitch_events(self,mf):
        events = []
        for _ in tqdm(range(self.n_events),desc='Stitching events',leave=False):
            N = min(20,max(1,np.random.poisson(self.mean_poisson)))
            min_dist = -1.
            n_hits = 0
            while min_dist < self.min_dist or n_hits == 0:
                indices = np.random.choice(len(mf), size=N, replace=False)
                mfs_to_stich = [
                    MiniFrame(
                        {
                            key : mf[key][idx].reshape(1,-1)
                            for key in mf.keys()
                        }
                    )
                    for idx in indices
                ]
                if N ==1:
                    min_dist = math.inf
                else:
                    distances = self.get_distances(mfs_to_stich)
                    min_dist = distances.min()
                event = self.stitch_one_event(mfs_to_stich)
                n_hits = event['hits']['pos'].shape[0]
            events.append(event)

        return CaloGraphDataset.from_data_list(events)


if __name__ == "__main__":
    config_file_path = sys.argv[1]
    parameter_dict_file_path = sys.argv[2]
    output_path = sys.argv[3]

    with open(config_file_path, "r") as file:
        config = json.load(file)['simulation']
    with open(parameter_dict_file_path, "r") as file:
        parameter_dict = json.load(file)

    generator = Simulation(config,parameter_dict)
    mf = generator.run_simulation()
    dataset = generator.stitch_events(mf)
    dataset.save(output_path)
