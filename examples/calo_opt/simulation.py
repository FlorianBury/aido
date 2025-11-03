import json
import os
import sys
import random
import itertools

import numpy as np
import pandas as pd
from G4Calo import GeometryDescriptor, run_batch
from minipandas import MiniFrame,concat

class Simulation():
    def __init__(
            self,
            parameter_dict: dict
            ):
        self.parameter_dict = parameter_dict

        if "num_events" in parameter_dict:
            self.n_events = parameter_dict["num_events"]["current_value"]
        else:
            self.n_events = 100
        if "N_max_gamma" in parameter_dict:
            self.N_max_gamma = parameter_dict["N_max_gamma"]["current_value"]
        else:
            self.N_max_gamma = 1
        if "N_max_pion" in parameter_dict:
            self.N_max_pion = parameter_dict["N_max_pion"]["current_value"]
        else:
            self.N_min_pion = 1
        if "N_min_gamma" in parameter_dict:
            self.N_min_gamma = parameter_dict["N_min_gamma"]["current_value"]
        else:
            self.N_min_gamma = 0
        if "N_min_pion" in parameter_dict:
            self.N_min_pion = parameter_dict["N_min_pion"]["current_value"]
        else:
            self.N_min_pion = 0
        if "minEnergy_GeV" in parameter_dict:
            self.minEnergy_GeV = max(1e-3,parameter_dict["minEnergy_GeV"]["current_value"])
        else:
            self.minEnergy_GeV = 1.
        if "maxEnergy_GeV" in parameter_dict:
            self.maxEnergy_GeV = max(self.minEnergy_GeV,parameter_dict["maxEnergy_GeV"]["current_value"])
        else:
            self.maxEnergy_GeV = min(self.minEnergy_GeV,20.)

        self.cw = GeometryDescriptor()

        for i in range(3):
            self.cw.addLayer(
                max(parameter_dict[f"thickness_absorber_{i}"]["current_value"], 1e-3),
                parameter_dict[f"material_absorber_{i}"]["current_value"],
                False,
                1
            )
            self.cw.addLayer(
                max(parameter_dict[f"thickness_scintillator_{i}"]["current_value"], 1e-3),
                parameter_dict[f"material_scintillator_{i}"]["current_value"],
                True,
                1
            )

    def run_simulation(self) -> pd.DataFrame:
        mfs = []
        rng = np.random.default_rng(seed=self.parameter_dict["metadata"]["rng_seed"])
        N_gammas = []
        N_pions = []
        for i,(Ng,Np) in enumerate(
                itertools.product(
                    np.arange(self.N_max_gamma),
                    np.arange(self.N_max_pion),
                )
        ):
            if Ng+Np == 0:
                continue
            mf: MiniFrame = run_batch(
                gd=self.cw,
                nEvents=self.n_events,
                particleSpec=['gamma']*Ng+['pi+']*Np,
                minEnergy_GeV=self.minEnergy_GeV*(Ng+Np),
                maxEnergy_GeV=self.maxEnergy_GeV*(Ng+Np),
                no_mp=True,
                manual_seed=self.parameter_dict["metadata"]["rng_seed"]+i,
            )
            N_gammas.append(np.full(len(mf),Ng,dtype='float32'))
            N_pions.append(np.full(len(mf),Np,dtype='float32'))
            mfs.append(mf)
        df = concat(mfs, axis=0, ignore_index=True).to_pandas(indiv_cols=False)
        df = df.assign(N_gamma=np.concatenate(N_gammas,axis=0))
        df = df.assign(N_pion=np.concatenate(N_pions,axis=0))
        return df


if __name__ == "__main__":
    parameter_dict_file_path = sys.argv[1]
    output_path = sys.argv[2]

    with open(parameter_dict_file_path, "r") as file:
        parameter_dict = json.load(file)

    generator = Simulation(parameter_dict)
    df = generator.run_simulation()

    for column in df.columns:
        if df[column].dtype == "awkward":
            df[column] = df[column].to_list()

    df.to_parquet(output_path)
    os.system("rm -f ./*.pkl")
