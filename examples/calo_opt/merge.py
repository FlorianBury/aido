import os
import re
import sys
import json
import pickle
import numpy as np
import pandas as pd
from minipandas import MiniFrame,concat

def super_concat(mfs):
    mf = SuperMiniFrame()
    for _mf in mfs:
        for key in _mf.keys():
            mf.add(key,_mf[key])
    return mf


class SuperMiniFrame:
    def __init__(self,data=None):
        if data is None:
            self._data = {}
        else:
            self._data = data

    def add(self,key,mf):
        if key in self._data.keys():
            self._data[key] = concat([self._data[key],mf],axis=0,ignore_index=True)
        else:
            self._data[key] = mf

    def __getitem__(self,key):
        assert key in self.keys(), f'No {key} in {self.keys()}'
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(list(self._data.values())[0])

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def to_pickle(self, path: str, protocol: int = 4):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    key: mf._data
                    for key, mf in self._data.items()
                },
                f,
                protocol=protocol,
            )

    @staticmethod
    def read_pickle(path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return SuperMiniFrame(
            {
                key : MiniFrame(values)
                for key, values in data.items()
            }
        )

    def to_pandas(self):
        return pd.concat({key:self[key].to_pandas() for key in self.keys()},axis=1)

class Merge:
    def __init__(self,config_path):
        with open(config_path,'r') as handle:
            config = json.load(handle)
        self.link = {
            'Inputs' : [
                'sensor_energy',
                'sensor_x',
                'sensor_y',
                'sensor_z',
                'sensor_dx',
                'sensor_dy',
                'sensor_dz',
                'sensor_layer'
            ],
            'Targets' : config['reconstruction']['targets'],
            'Classes' : config['classification']['classes'],
            'Context' : [
            ],
            'Parameters' : [
            ],
        }
        self.mfs = []

    def convert_sim_to_reco(
        self,
        parameter_values_file_path: str,
        simulation_file_path: str,
    ):
        # Miniframe #
        mf = MiniFrame.read_pickle(simulation_file_path)

        # Parameters #
        with open(parameter_values_file_path,'r') as handle:
            parameter_values = json.load(handle)

        # Add parameters to miniframe #
        for key,val in parameter_values.items():
            assert key not in mf.keys()
            mf._data[key] = np.ones(len(mf)) * val
            if key not in self.link['Parameters']:
                self.link['Parameters'].append(key)

        # Record mf #
        self.mfs.append(mf)


    def write(self,reco_output_path):
        # Concat mfs #
        print (f'Concatenating {len(self.mfs)} miniframes')
        mf = concat(self.mfs)

        # Make super miniframe #
        self.super_mf = SuperMiniFrame()
        for super_key, keys in self.link.items():
            if len(keys) > 0:
                self.super_mf.add(
                    key = super_key,
                    mf = MiniFrame(
                        data = {
                            key : mf[key]
                            for key in keys
                        }
                    )
                )

        # Save #
        print (f'Saving as {reco_output_path}')
        self.super_mf.to_pickle(reco_output_path)


if __name__ == "__main__":
    config_path = sys.argv[1]
    reco_input_path = sys.argv[2]
    paths = sys.argv[3:]
    assert len(paths) % 2 == 0
    N = int(len(paths)/2)
    parameter_values_file_paths = paths[:N]
    simulation_file_paths = paths[N:]

    merge = Merge(config_path)
    for idx,(parameter_values_file_path,simulation_file_path) in enumerate(zip(parameter_values_file_paths,simulation_file_paths)):
        print (f'... {simulation_file_path} [{idx+1}/{len(simulation_file_paths)}]')
        merge.convert_sim_to_reco(
            parameter_values_file_path,
            simulation_file_path,
        )

    merge.write(reco_input_path)

    # Hack to remove a simulation file after the merge to avoid keeping same data twice
    # While leaving an empty file (otherwise b2luigi thinks the task must be resubmitted)
    for file in simulation_file_paths:
        os.remove(file)
        with open(file,'a'):
            os.utime(file,None)
    print ('Removed Simulation files')

    os.system("rm -f ./*.pkl")
