import sys
import os
import numpy as np 
from pathlib import Path
from math import log
from secmlt.metrics.classification import Accuracy
from torch.utils.data import TensorDataset, DataLoader

from maltorch.adv.evasion.base_optim_attack_creator import OptimizerBackends
from maltorch.adv.evasion.zexe import ZEXESectionInjection
from maltorch.data.loader import load_from_folder, create_labels, load_single_exe
from maltorch.zoo.avaststyleconv import AvastStyleConv
from maltorch.zoo.bbdnn import BBDnn
from maltorch.zoo.ember_gbdt import EmberGBDT
from maltorch.zoo.malconv import MalConv

from maltorch.optim.zexe import ExplorationStrategyID


device = "cpu"

malware_folder = Path(sys.argv[1])

out_dir = sys.argv[2]
model_name = sys.argv[3]
num_sections = int(sys.argv[4])
content_size = int(sys.argv[5])
os.makedirs(out_dir, exist_ok=True)

file_paths = [path for path in malware_folder.glob("*.dat")][5:]


expl_start = ExplorationStrategyID.LANGEVIN
expl_strat_params = {'beta' : lambda t : np.log(t + 2)/ t ,  'seed' : 123123123}
#expl_strat_params = {'perturbation_size' : (content_size + 8) * num_sections,'popsize' : 10,  'seed' : 123123123}


networks = {
   'malconv': MalConv.create_model(),
   'avast_style_conv': AvastStyleConv.create_model(),
   'ember_gbdt' : EmberGBDT.create_model(),
   'bbdnn': BBDnn.create_model(),
}
model = networks[model_name]
reps = 10
for r in range(reps):
#    for (model_name, model) in networks.items():

    with open(f"{out_dir}/zexe_{model_name}_{num_sections}_{r}.txt", "w") as f:
        for i in range(len(file_paths)):

            #             query_budget: int,
            # perturbation_size: int,
            # how_many_sections: int,
            # num_workers: int = 1,
            # stepsize: float = 1.0, 
            # h : float = 1.0,
            # num_directions : int = 1,
            # armijo_constant : float = 1e-5,
            # min_stepsize : float = 1e-5,
            # max_stepsize : float = 1e3,
            # contraction_factor : float = 0.5,
            # expansion_factor : float = 2.0,
            # exploration_strategy: ExplorationStrategyID = ExplorationStrategyID.RANDOM,
            # expl_strategy_params: dict = {},     

            zexe_attack = ZEXESectionInjection(
                query_budget = 500,
                perturbation_size = (content_size + 8) * num_sections,# * 100,
                how_many_sections = num_sections,#00,
                num_directions = 5, # aka population size
                min_stepsize = 1.0,
                max_stepsize = 1000.0,
                armijo_constant = 1.0,
                contraction_factor = 0.5,
                expansion_factor = 2.0,
                exploration_strategy = expl_start,
                expl_strategy_params = expl_strat_params,
                stepsize =10.0,
                h=100000.0
            )

            print(f"[{i}/{len(file_paths)}] Attacking {file_paths[i]}")
            x = load_single_exe(file_paths[i]).reshape(1, -1).long()
            y = create_labels(x, 1, device=device)
            dl = DataLoader(TensorDataset(x, y), batch_size=1)
            pre_accuracy, pre_score = Accuracy()(model, dl), model(x).item()
            print("- - - Pre-attack accuracy: ", Accuracy()(model, dl).item(), pre_score, model.threshold)
            adv_dl = zexe_attack(model, dl)
            accuracy = Accuracy()(model, adv_dl)
            print("Accuracy: ", accuracy)
            plain_length = x.flatten().shape[0]
            for (x_adv, y_adv) in adv_dl:
                score =  model(x_adv).item()
                adv_length = x_adv.flatten().shape[0]
                print("Score: ", score, y_adv)
                print("ADDED BYTES", x.flatten().shape[0], x_adv.flatten().shape[0])
            f.write(f"{str(file_paths[i]).split('/')[-1]},{pre_accuracy},{pre_score},{accuracy},{score},{plain_length},{adv_length}\n")
            f.flush()
