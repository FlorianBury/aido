import json
import os
import sys
import random
from tqdm import tqdm
import numpy as np
import torch

from torch_geometric.data import Data

import numpy as np
import pandas as pd
from G4Calo import GeometryDescriptor, run_batch
from minipandas import MiniFrame,concat

from dataset import CaloGraphDataset

class Simulation():
    def __init__(
            self,
            parameter_dict: dict
        ):
        self.parameter_dict = parameter_dict
        self.particle_types = [
            "e-",
            "e+",
            "gamma",
            #"pi+",
            #"pi-",
            #"pi0",
            #"proton",
            #"neutron",
        ]
        if "number_simulations_per_particle" in parameter_dict:
            self.n_parts = parameter_dict["number_simulations_per_particle"]["current_value"]
        else:
            self.n_parts = 100
        if "number_stitched_events" in parameter_dict:
            self.n_events = parameter_dict["number_stitched_events"]["current_value"]
        else:
            self.n_events = 100
        if "mean_number_particles_per_event" in parameter_dict:
            self.mean_poisson = parameter_dict["mean_number_particles_per_event"]["current_value"]
        else:
            self.mean_poissoni = 1
        if "min_distance_cut" in parameter_dict:
            self.min_dist = parameter_dict["min_distance_cut"]["current_value"]
        else:
            self.min_dist = 0.
        if "minEnergy_GeV" in parameter_dict:
            self.minEnergy_GeV = max(1e-3,parameter_dict["minEnergy_GeV"]["current_value"])
        else:
            self.minEnergy_GeV = 1.
        if "maxEnergy_GeV" in parameter_dict:
            self.maxEnergy_GeV = max(self.minEnergy_GeV,parameter_dict["maxEnergy_GeV"]["current_value"])
        else:
            self.maxEnergy_GeV = min(self.minEnergy_GeV,20.)

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
        x = torch.from_numpy(mfs[0]['sensor_x'])
        y = torch.from_numpy(mfs[0]['sensor_y'])
        z = torch.from_numpy(mfs[0]['sensor_z'])
        dx = torch.from_numpy(mfs[0]['sensor_dx'])
        dy = torch.from_numpy(mfs[0]['sensor_dy'])
        dz = torch.from_numpy(mfs[0]['sensor_dz'])
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
        labels[fracs.max(dim=0).values < 0.05] = -1

        # Make mask or non-zero cells and save #
        mask = E > 0
        event = Data(
            # Calo info #
            pos = torch.stack(
                [x[mask],y[mask],z[mask]],
                axis = -1,
            ).to(torch.float32),
            E = E[mask].unsqueeze(-1).to(torch.float32),
            layer = layer[mask].unsqueeze(-1).to(torch.float32),
            cell = torch.stack(
                [dx[mask],dy[mask],dz[mask]],
                axis = -1,
            ).to(torch.float32),
            labels = labels[mask].unsqueeze(-1).to(torch.int64),
            # Particle info #
            particle_pos = torch.stack(
                [part_x,part_y],
                axis = -1,
            ).to(torch.float32),
            particle_E = part_E.unsqueeze(-1).to(torch.float32),
            particle_type = part_type.to(torch.float32),
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
            while min_dist < self.min_dist:
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
                if N == 1:
                    break
                distances = self.get_distances(mfs_to_stich)
                min_dist = distances.min()
                if min_dist > self.min_dist:
                    N = max(1,N-1)
            events.append(self.stitch_one_event(mfs_to_stich))

        return CaloGraphDataset.from_data_list(events)


if __name__ == "__main__":
    parameter_dict_file_path = sys.argv[1]
    output_path = sys.argv[2]

    with open(parameter_dict_file_path, "r") as file:
        parameter_dict = json.load(file)

    generator = Simulation(parameter_dict)
    mf = generator.run_simulation()
    dataset = generator.stitch_events(mf)
    dataset.save(output_path)
