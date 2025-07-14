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

device = "cpu"

malware_folder = Path(sys.argv[1])

out_dir = sys.argv[2]

model_name = sys.argv[3]
num_sections = int(sys.argv[4])
os.makedirs(out_dir, exist_ok=True)

budget = 1000
file_paths = [path for path in malware_folder.glob("*.dat")]



networks = {
   'malconv': MalConv.create_model(),
   'avast_style_conv': AvastStyleConv.create_model(),
   'ember_gbdt' : EmberGBDT.create_model(),
   'bbdnn': BBDnn.create_model(),
}
model = networks[model_name]
reps = 10
for r in range(reps):

    with open(f"{out_dir}/gamma_{model_name}_{r}.txt", "w") as f:
        for i in range(len(file_paths)):
            attack = GAMMASectionInjection(
                query_budget=budget,
                benignware_folder= Path("/data/mrando/goodwares/goodware"),#exe_folder / ".." / "benignware",
                which_sections=[".text"],
                how_many_sections=num_sections
            #    device=device
            )

            print(f"[{i}/{len(file_paths)}] Attacking {file_paths[i]}")
            x = load_single_exe(file_paths[i]).reshape(1, -1).long()
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
            f.write(f"{str(file_paths[i]).split('/')[-1]},{pre_accuracy},{pre_score},{accuracy},{score},{plain_length},{adv_length}\n")
            f.flush()
