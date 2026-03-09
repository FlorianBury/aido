import os
from typing import Dict

import torch
from calo_opt.interface import CaloOptInterface  # Import your derived class
from calo_opt.plotting import CaloOptPlotting
from calo_opt.config import CaloConfig

import aido

class UIFullCalorimeter(CaloOptInterface):

    @classmethod
    def constraints(
            self,
            parameter_dict: aido.SimulationParameterDictionary,
            parameter_dict_as_tensor: Dict[str, torch.Tensor]
            ) -> torch.Tensor:
        """ Additional constraints to add to the Loss.

        Use the Tensors found in 'parameter_dict_as_tensor' (Dict) to compute the constraints
        and return a 1-dimensional Tensor. Note that missing gradients at this stage will
        negatively impact the training of the optimizer.

        Use the usual 'parameter_dict' instance to access additional information such as
        boundaries, costs per item and all other stored values.

        In this example, we add the cost per layer for all six layers by looping over the
        index of the layer (0, 1, 2) and their type (absorber / scintillator). Using the

        """
        detector_length_list = []
        cost_list = []
        num_layers = parameter_dict['num_layers'].current_value

        for i in range(num_layers):
            for name in ["absorber", "scintillator"]:
                material_probabilities = parameter_dict_as_tensor[f"material_{name}:{i}"]
                material_cost = torch.tensor(
                    parameter_dict[f"material_{name}:{i}"].cost,
                    device=material_probabilities.device
                )
                layer_weighted_cost = material_probabilities * material_cost
                layer_thickness = parameter_dict_as_tensor[f"thickness_{name}:{i}"]

                cost_list.append(layer_thickness * layer_weighted_cost)
                detector_length_list.append(layer_thickness)

        max_length = parameter_dict["max_length"].current_value
        max_cost = parameter_dict["max_cost"].current_value
        detector_length = torch.stack(detector_length_list).sum()
        cost = torch.stack(cost_list).sum()
        detector_length_penalty = torch.mean(torch.nn.functional.relu((detector_length - max_length) / max_length)**2)
        max_cost_penalty = torch.mean(torch.nn.functional.relu((cost - max_cost) / max_cost)**2)
        return detector_length_penalty + max_cost_penalty

    def plot(self, parameter_dict: aido.SimulationParameterDictionary) -> None:
        calo_opt_plotter = CaloOptPlotting(self.results_dir)
        calo_opt_plotter.mplstyle()
        calo_opt_plotter.plot()
        return None


if __name__ == "__main__":
    num_layers: int = 5

    ui_interface = UIFullCalorimeter(CaloConfig())
    ui_interface.container_path = "/cephfs/dice/users/lw23382/sif/minicalosim_90baa21.sif"
    ui_interface.container_extra_flags = ""
    #ui_interface.container_extra_flags = "-B /software,/cephfs"
    ui_interface.verbose = True
    results_dir: str = "/cephfs/dice/users/lw23382/AIDO/aido_binary_reg_lambda_2_v1"

    # Non optimizable #
    parameters = [
        aido.SimulationParameter("num_layers", num_layers, optimizable=False),
        aido.SimulationParameter("max_length", 100, optimizable=False),
        aido.SimulationParameter("max_cost", 100_000, optimizable=False),
        aido.SimulationParameter("num_events", 500, optimizable=False),
        aido.SimulationParameter("granularity:0", 25, optimizable=False),
        aido.SimulationParameter("granularity:1", 25, optimizable=False),
        aido.SimulationParameter("granularity:2", 25, optimizable=False),
        aido.SimulationParameter("granularity:3", 25, optimizable=False),
        aido.SimulationParameter("granularity:4", 25, optimizable=False),
            # num_events is now per batch of a set of number of particles
        #aido.SimulationParameter(
        #    name = "N:pi+",
        #    starting_value = 0,
        #    min_value = 0,
        #    max_value = 1,
        #    optimizable = False,
        #),
        #aido.SimulationParameter(
        #    name = "N:proton",
        #    starting_value = 0,
        #    min_value = 0,
        #    max_value = 1,
        #    optimizable = False,
        #),
        aido.SimulationParameter(
            name = "N:e+",
            starting_value = 0,
            min_value = 0,
            max_value = 1,
            optimizable = False,
        ),
        aido.SimulationParameter(
            name = "N:gamma",
            starting_value = 0,
            min_value = 0,
            max_value = 1,
            optimizable = False,
        ),
        aido.SimulationParameter("minEnergy_GeV", 5., optimizable=False),
        aido.SimulationParameter("maxEnergy_GeV", 100., optimizable=False),
        aido.SimulationParameter("sharedEnergy", False, optimizable=False),
        aido.SimulationParameter("exclusiveSimulation", True, optimizable=False),
    ]
    # Layers #
    for i in range(num_layers):
        parameters.append(
            aido.SimulationParameter(f"thickness_absorber:{i}", 10., min_value=0., sigma=2.5),
        )
        parameters.append(
            aido.SimulationParameter(f"thickness_scintillator:{i}", 3., min_value=0., sigma=2.5)
        )
        parameters.append(
            aido.SimulationParameter(
                f"material_absorber:{i}",
                "G4_Fe",
                discrete_values=["G4_Pb", "G4_Fe"],
                cost=[25, 4.166],
                probabilities=[0.5, 0.5],
            )
        )
        parameters.append(
            aido.SimulationParameter(
                f"material_scintillator:{i}",
                "G4_POLYSTYRENE",
                discrete_values=["G4_PbWO4", "G4_POLYSTYRENE"],
                cost=[2500.0, 0.01],
                probabilities=[0.5, 0.5],
            )
        )
    # Optimize #
    aido.optimize(
        parameters=aido.SimulationParameterDictionary(parameters),
        user_interface=ui_interface,
        simulation_tasks=20,
        max_iterations=100,
        threads=20,
        results_dir=results_dir,
        description="""
Optimization of a sampling calorimeter with cost and length constraints.
Includes the optimization of discrete parameters and specific plotting functions
"""
    )
    os.system("rm *.root")
