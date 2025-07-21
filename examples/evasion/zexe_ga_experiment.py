import sys
import os
import re
import torch
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
import multiprocessing
from multiprocessing import Pool
import itertools

device = "cpu"

malware_folder = Path(sys.argv[1])

out_dir = sys.argv[2]
gamma_result_folder = sys.argv[3]
model_name = sys.argv[4]
num_sections = int(sys.argv[5])
fraction_of_gamma_perturbation = float(sys.argv[6])
threshold = float(sys.argv[7])
num_processes = int(sys.argv[8])


print("FRACTION: ", fraction_of_gamma_perturbation)

os.makedirs(out_dir, exist_ok=True)

file_paths = [path for path in malware_folder.glob("*.dat")]
reps_list = list(range(5))

#expl_strat_params = {'perturbation_size' : (content_size + 8) * num_sections,'popsize' : 10,  'seed' : 123123123}


def create_model(name, threshold):
    if name == "malconv":
        return MalConv.create_model(threshold=threshold)
    elif name == "avast_style_conv":
        return AvastStyleConv.create_model(threshold=threshold)
    elif name == "ember_gbdt":
        return EmberGBDT.create_model(threshold=threshold)
    elif name == "bbdnn":
        return BBDnn.create_model(threshold=threshold)
    raise ValueError(f"Unknown model name: {name}")

regex = re.compile(r"gamma_[a-zA-Z0-9]+_({})_({})_(\d)".format(model_name, num_sections))

global_psize = []
for file in os.listdir(gamma_result_folder):
    if regex.match(file):
        with open(f"{gamma_result_folder}/{file}", "r") as f:
            line = f.readline().split(",")
            if len(line) > 1:
                plain_length = int(line[4])
                adv_length = int(line[5])
                global_psize.append(adv_length - plain_length)
global_psize = int(np.mean(global_psize))
# print("[--] Average perturbation size: ", np.mean(avg_pert_size), np.std(avg_pert_size))
# perturbation_size = int(np.mean(avg_pert_size))
#exit()
reps = 5
model = create_model(model_name, threshold) 



def run_experiment(fpath_rep):
    file_path, r = fpath_rep
    
    perturbation_size = []
    for i in range(5):
        if os.path.exists(f"{gamma_result_folder}/gamma_{str(file_path).split('/')[-1].split('.')[0]}_{model_name}_{num_sections}_{i}.txt"):
            with open(f"{gamma_result_folder}/gamma_{str(file_path).split('/')[-1].split('.')[0]}_{model_name}_{num_sections}_{i}.txt", "r") as f:
                line = f.readline().split(",")
                if len(line) > 1:
                    plain_length = int(line[4])
                    adv_length = int(line[5])
                    perturbation_size.append(adv_length - plain_length)

    perturbation_size = int(np.mean(perturbation_size)) if len(perturbation_size) > 0 else global_psize


    content_size = (int(perturbation_size * fraction_of_gamma_perturbation) // num_sections) - 8
    expl_start = ExplorationStrategyID.GA
    expl_strat_params = {'eps' : 0.01, 'perturbation_size' : int((content_size + 8) * num_sections), 'popsize' : 10,  'seed' : 1373 + 123*r}


    
    # for r in range(reps):
    torch.manual_seed(1123 + 143*r)
    attack = ZEXESectionInjection(
                query_budget = 500,
                perturbation_size =  int((content_size + 8) * num_sections),# * 100,
                how_many_sections = num_sections,#00,
                num_directions = 5, # aka population size
                min_stepsize = 0.1,
                max_stepsize = 1000.0,
                armijo_constant = 1e-5,
                contraction_factor = 0.5,
                expansion_factor = 2.0,
                exploration_strategy = expl_start,
                expl_strategy_params = expl_strat_params,
                stepsize =1.0,
                h=100000.0,
                seed = 1841 + 201*r
            )

    print(f"[{r}/{reps}] Attacking {file_path}...")
    print("[--] GAMMA PERTURBATION SIZE: ", perturbation_size)
    print("[--] ZEXE PERTURBATION SIZE: ", int((content_size + 8) * num_sections))
    with open(f"{out_dir}/zexe_ga_{str(file_path).split('/')[-1].split('.')[0]}_{model_name}_{num_sections}_{r}.txt", "a") as f:
        x = load_single_exe(file_path).reshape(1, -1).long()
        y = create_labels(x, 1, device=device)
        dl = DataLoader(TensorDataset(x, y), batch_size=1)
        pre_accuracy, pre_score = Accuracy()(model, dl), model(x).item()
        print("- - - Pre-attack accuracy: ", Accuracy()(model, dl).item(), pre_score, model.threshold)
        adv_dl = attack(model, dl)
        accuracy = Accuracy()(model, adv_dl)
        plain_length = x.flatten().shape[0]
        for (x_adv, y_adv) in adv_dl:
            score =  model(x_adv).item()
            adv_length = x_adv.flatten().shape[0]
            print("Score: ", score, y_adv)
            print("ADDED BYTES", x_adv.flatten().shape[0] - x.flatten().shape[0])
        f.write(f"{pre_accuracy},{pre_score},{accuracy},{score},{plain_length},{adv_length}\n")
        f.flush()


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    with Pool(processes=num_processes) as p:

        p.map(run_experiment, list(itertools.product(file_paths, reps_list)))

