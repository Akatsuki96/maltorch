from pathlib import Path

from secmlt.metrics.classification import Accuracy
from torch.utils.data import TensorDataset, DataLoader

from maltorch.adv.evasion.gamma_section_injection import GAMMASectionInjection
from maltorch.data.loader import load_from_folder, create_labels
from maltorch.zoo.avaststyleconv import AvastStyleConv
from maltorch.zoo.bbdnn import BBDnn
from maltorch.zoo.ember_gbdt import EmberGBDT
from maltorch.zoo.malconv import MalConv
#device = "mps"

exe_folder = Path("/data/mrando/malwares/malware_testset") #Path(__file__).parent / ".." / "data" / "malware"
X = load_from_folder(exe_folder, "dat", device='cpu', padding=255, limit=100)[3:4]
y = create_labels(X, 1, device='cpu')
dl = DataLoader(TensorDataset(X, y), batch_size=1)

attack = GAMMASectionInjection(
    query_budget=50,
    benignware_folder= Path("/data/mrando/goodwares/goodware"),#exe_folder / ".." / "benignware",
    which_sections=[".text"],
    how_many_sections=10
#    device=device
)
model = BBDnn.create_model()#MalConv.create_model()

print("Pre-attack accuracy: ", Accuracy()(model, dl))
adv_dl = attack(model, dl)
print("Accuracy: ", Accuracy()(model, adv_dl))

for (x, y) in adv_dl:
    print((x.flatten().shape[0] - X[0].flatten().shape[0]) / X[0].flatten().shape[0])  
