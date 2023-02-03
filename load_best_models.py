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
        w_norm = torch.linalg.norm(lin_model.weight.data, ord=np.inf).item()
        b_norm = torch.linalg.norm(lin_model.bias.data, ord=np.inf).item()
        norm = max([w_norm, b_norm])
        print(norm)
        print(w_norm, b_norm)
        for j in range(shape[1]):
            for i in range(shape[0]):
                vars.append(lin_model.weight[j][i].data.item()/norm)
            vars.append(lin_model.bias[j].item()/norm)

        print(vars)
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
