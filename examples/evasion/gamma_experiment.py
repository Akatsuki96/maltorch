import sys
import os
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
from maltorch.adv.evasion.gamma_section_injection import GAMMASectionInjection

from maltorch.optim.zexe import ExplorationStrategyID

from multiprocessing import Pool

device = "cpu"

malware_folder = Path(sys.argv[1])

out_dir = sys.argv[2]
goodware_dir = sys.argv[3]

threshold = float(sys.argv[4])

model_name = sys.argv[5]
num_sections = int(sys.argv[6])

num_processes = int(sys.argv[7])

os.makedirs(out_dir, exist_ok=True)

budget = 500
file_paths = [path for path in malware_folder.glob("*.dat")]






networks = {
   'malconv': MalConv.create_model(threshold=threshold),
   'avast_style_conv': AvastStyleConv.create_model(threshold=threshold),
   'ember_gbdt' : EmberGBDT.create_model(threshold=threshold),
   'bbdnn': BBDnn.create_model(threshold=threshold),
}
model = networks[model_name]
reps = 5
def run_experiment(file_path):
    for r in range(reps):
        print(f"[{r}/{reps}] Attacking {file_path}...")
        with open(f"{out_dir}/gamma_{str(file_path).split('/')[-1].split('.')[0]}_{model_name}_{num_sections}.txt", "w") as f:
            attack = GAMMASectionInjection(
                query_budget=budget,
                benignware_folder= Path(goodware_dir),#exe_folder / ".." / "benignware",
                which_sections=[".text"],
                how_many_sections=num_sections
            )
            x = load_single_exe(file_path).reshape(1, -1).long()
            y = create_labels(x, 1, device=device)
            dl = DataLoader(TensorDataset(x, y), batch_size=1)
            pre_accuracy, pre_score = Accuracy()(model, dl), model(x).item()
            print("- - - Pre-attack accuracy: ", Accuracy()(model, dl).item(), pre_score, model.threshold)
            adv_dl = attack(model, dl)
            accuracy = Accuracy()(model, adv_dl)
            print("Accuracy: ", accuracy)
            plain_length = x.flatten().shape[0]
            for (x_adv, y_adv) in adv_dl:
                score =  model(x_adv).item()
                adv_length = x_adv.flatten().shape[0]
                print("Score: ", score, y_adv)
                print("ADDED BYTES", x.flatten().shape[0], x_adv.flatten().shape[0])
            f.write(f"{pre_accuracy},{pre_score},{accuracy},{score},{plain_length},{adv_length}\n")
            f.flush()

with Pool(processes=num_processes) as p:
    p.map(run_experiment, file_paths)

