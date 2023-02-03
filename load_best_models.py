import os
import torch
import torch.nn as nn
import torch.linalg
import utils
import numpy as np

os.chdir("./examples/logs")


def load_best_model(dataset, algo, run, seed):
    run_dir = os.path.join(dataset, algo, "run_"+str(run)+"_seed_"+str(seed))
    model_file = os.path.join(run_dir, dataset+"_seed_"+str(seed)+"_epoch_last_model.pth")
    return model_file

def get_weight_var(dataset, algo, runs=range(1,7), seeds=range(10,16), shape=(2,2)):
    max_vars = []
    for run, seed in zip(runs, seeds):
        model_file = load_best_model(dataset, algo, run, seed)
        lin_model = nn.Linear(shape[0], shape[1])
        utils.load(lin_model, model_file, device=torch.device('cpu'))
        vars = []
        for j in range(shape[1]):
            for i in range(shape[0]):
                vars.append(lin_model.weight[j][i].data.item())
            vars.append(lin_model.bias[j].item())
        max_vars.append(max(vars))
    return np.var(max_vars)


for dataset in ["noisy_2feature", "rot_simple", "spu_2feature"]:
    for algo in ["CG", "groupDRO"]:
        print("Dataset: " + dataset + ", algorithm: "+algo)
        if dataset=="spu_2feature":
            var = get_weight_var(dataset, algo, shape=(3,2))
        else:
            var = get_weight_var(dataset, algo)
        print("Variance:", var)

for dataset in ["noisy_5group", "rot_5group", "spu_5group"]:
    for algo in ["CG", "groupDRO"]:
        print("Dataset: " + dataset + ", algorithm: "+algo)
        if dataset=="spu_5group":
            var = get_weight_var(dataset, algo, shape=(3,2), runs=range(301,307), seeds=range(10,16))
        else:
            var = get_weight_var(dataset, algo, runs=range(301,307), seeds=range(10,16))
        print("Variance:", var)