import numpy as np
import os 

algorithms = ['gamma', 'zexe_lan']#['gamma'] #['rs', 'zexe', 'gamma', 'zexers', 'zexe_lan', 'zexe_ga']
datasets = ['malconv']

def get_results(algorithm, dataset, num_sections):
    evasion = []
    added_bytes = []
    files = os.listdir(f"results/{dataset}/{algorithm}")
#    files = os.listdir(f"test/")
    for file in files:        
        if int(file.split("_")[-2]) != num_sections:# or int(file.split("_")[-1].split(".")[0]) != 0:
            continue
#        print(file)
        with open(f"results/{dataset}/{algorithm}/{file}", "r") as f:
#        with open(f"test/{file}", "r") as f:
            for line in f.readlines():
                splitted = line.split(",")
                if float(splitted[0]) == 0:
                    continue
#                if 1 - float(splitted[2]) == 1.0:
#                    print("BREAK!")
#                    print(splitted[1], splitted[3], file)

                evasion.append(1 - float(splitted[2]))
                added_bytes.append(float(splitted[5]) - float(splitted[4]) )
#                if len(evasion) == 91:
#                    return np.mean(evasion), np.mean(added_bytes), np.std(added_bytes), len(evasion)
    return np.mean(evasion), np.mean(added_bytes), np.std(added_bytes), len(evasion)

for num_sections in [30]:#, 40, 50]:
    for algorithm in algorithms:
        for dataset in datasets:
            evasion_rate, mu_len, std_len, num_samples = get_results(algorithm, dataset, num_sections)
            print(f"{dataset}[{algorithm}]:\n\t[--] Evasion Rate {evasion_rate:.4f}\n\t[--] Mean Length {mu_len:.4f} +/- {std_len:.4f}\n\t[--] Number of samples {num_samples}")