import numpy as np
import os 

algorithms = ['gamma', 'zexe_ga']#['gamma'] #['rs', 'zexe', 'gamma', 'zexers', 'zexe_lan', 'zexe_ga']
datasets = ['ember_gbdt']

def get_results(algorithm, dataset, num_sections):
    evasion = []
    added_bytes = []
    files = os.listdir(f"/mnt/data/mrando/zexe_results/{dataset}/{algorithm}")
    valid_files= []
    for file in os.listdir("/mnt/data/mrando/zexe_results/ember_gbdt/zexe_ga"):
        if int(file.split("_")[-2]) == num_sections and int(file.split("_")[-1].split(".")[0]) == 0:
            with open(f"/mnt/data/mrando/zexe_results/ember_gbdt/zexe_ga/{file}", 'r') as f:
                if len(f.readlines()) == 0:
                    continue
            valid_files.append(file.split("_")[-5])
#    files = os.listdir(f"test/")
 #   print(valid_files)
    for file in files:        
        if int(file.split("_")[-2]) != num_sections or file.split("_")[-5] not in valid_files or int(file.split("_")[-1].split(".")[0]) != 0:
            continue
#        print(file)
        with open(f"/mnt/data/mrando/zexe_results/{dataset}/{algorithm}/{file}", "r") as f:
#        with open(f"test/{file}", "r") as f:
#            print(file)
            for line in f.readlines()[:1]:
                splitted = line.split(",")
#                print(line)
#                if float(splitted[0]) == 0:
#                    continue
#                if 1 - float(splitted[2]) == 1.0:
#                    print("BREAK!")
#                    print(splitted[1], splitted[3], file)

                evasion.append(1 - float(splitted[2]))
                added_bytes.append(float(splitted[5]) - float(splitted[4]) )
#        print(len(evasion), evasion)
#                if len(evasion) == 91:
#                    return np.mean(evasion), np.mean(added_bytes), np.std(added_bytes), len(evasion)
    return np.mean(evasion), np.mean(added_bytes), np.std(added_bytes), len(evasion)

for num_sections in [30]:#, 40, 50]:
    for algorithm in algorithms:
        for dataset in datasets:
            evasion_rate, mu_len, std_len, num_samples = get_results(algorithm, dataset, num_sections)
            print(f"{dataset}[{algorithm}]:\n\t[--] Evasion Rate {evasion_rate:.4f}\n\t[--] Mean Length {mu_len:.4f} +/- {std_len:.4f}\n\t[--] Number of samples {num_samples}")
