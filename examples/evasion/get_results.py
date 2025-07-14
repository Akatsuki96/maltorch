import numpy as np


algorithms = ['rs', 'zexe', 'gamma', 'zexers', 'zexe_lan', 'zexe_ga']
datasets = ['malconv']

def get_results(algorithm, dataset):
    evasion = []
    added_bytes = []
    with open(f"results/{algorithm}_{dataset}_0.txt", "r") as f:
        for line in f.readlines():
            splitted = line.split(",")
            if float(splitted[1]) == 0:
                continue
            evasion.append(1 - float(splitted[3]))
            added_bytes.append(float(splitted[6]) - float(splitted[5]) )
    return np.mean(evasion), np.mean(added_bytes), np.std(added_bytes), len(evasion)


for algorithm in algorithms:
    for dataset in datasets:
        evasion_rate, mu_len, std_len, num_samples = get_results(algorithm, dataset)
        print(f"{dataset}[{algorithm}]:\n\t[--] Evasion Rate {evasion_rate:.4f}\n\t[--] Mean Length {mu_len:.4f} +/- {std_len:.4f}\n\t[--] Number of samples {num_samples}")