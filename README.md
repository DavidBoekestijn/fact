# Reproducibility of "Focus on the Common Good" 

Reproducibility, extensions and implementation of *Focus on the Common Good: Group Distributional Distribution Follows*. 

## Reproducibility
The implementation is modified to work with the newer version of the WILDS codebase, namely WILDS 2 instead of WILDS 1.2.2. These steps are to avoid errors to to add code that was not included in the original GitHub but was mentioned in the original paper.

### General
1. To avoid an attribute error, in [CG.py](examples/algorithms/CG.py) line 23 is commented out.
```python
# step size
# self.rho = config.metacg_rho
```

2. To avoid an EOF error, in [dataset.py](examples/configs/datasets.py) for all datasets with ```num_workers```, it is set to 0 instead of 1.
```python
'loader_kwargs': {
  'num_workers': 0,
  'pin_memory': True,
 }
 ```
 
 3. To avoid issues with the filenames, all ```:``` are changed to ```_``` in [utils.py](examples/utils.py), [run_expt.py](examples/run_expt.py) and [train.py](examples/train.py). So that it works on Mac-OS as well as on any other OS.
 
 ### Colored MNIST
 This datatset was not included in either the WILDS codebase or the implementation of the original authors.
 
1. Parameters for the C-MNIST dataset are added to [dataset.py](examples/configs/datasets.py).
```python
'cmnist': {
  'split_scheme': 'official',
  'model': 'resnet50',
  'model_kwargs': {'pretrained': True},
  'transform': 'image_base',
  'loss_function': 'cross_entropy',
  'groupby_fields': ['group', 'y'],
  'val_metric': 'acc_wg',
  'val_metric_decreasing': False,
  'optimizer': 'SGD',
  'optimizer_kwargs': {'momentum': 0.9},
  'scheduler': None,
  'batch_size': 64,
  'lr': 0.001,
  'weight_decay': 0.0,
  'n_epochs': 200,
  'algo_log_metric': 'accuracy',
  'process_outputs_function': 'multiclass_logits_to_pred',
}
```

2. A reference to this dataset is added in [get_dataset.py](wilds/get_dataset.py).
```python
elif dataset == 'cmnist':
  from wilds.datasets.cmnist_debug_dataset import CMNISTDDataset
  return CMNISTDDataset(**dataset_kwargs)
```

### Qualitative Analysis

 1. Create a folder with the dataset name in ```examples/data```.
 2. In [evaluate.py](examples/evaluate.py) add the following code in ```get_metrics()```.
 ```python
 elif "noisy_2feature" == dataset_name:
  return ["acc_avg"]
 elif "rot_simple" == dataset_name:
  return ["acc_avg"]
elif "spu_2feature" == dataset_name:
  return ["acc_avg"]
```

 3. Add the name of the datasets in ```additional_datasets = []``` in [init.py](wilds/__init__.py).
```python
'noisy_2feature',
'rot_simple',
'spu_2feature'
 ```
 
 4. Add references in [get_dataset.py](wilds/get_dataset.py).
```python
elif dataset == 'noisy_2feature':
  from wilds.datasets.noisy_simple_dataset import NoisySimpleDataset
  return NoisySimpleDataset(**dataset_kwargs)

elif dataset == 'spu_2feature':
  from wilds.datasets.spu_simple_dataset import SpuSimpleDataset
  return SpuSimpleDataset(**dataset_kwargs)

elif dataset == 'rot_simple':
  from wilds.datasets.rot_simple_dataset import RotSimpleDataset
  return RotSimpleDataset(**dataset_kwargs)
 ```
 
 5. Add parameters for the datasets in [datasets.py](examples/configs/datasets.py):
 ```python 
    'noisy_2feature': {
        'split_scheme': 'official',
        'model': 'logistic_regression',
        'loss_function': 'cross_entropy',
        'optimizer': 'SGD',
        'model_kwargs': {'in_features': 3},
        'algo_log_metric': 'accuracy',
        'val_metric': 'loss_all',
        'val_metric_decreasing': False,
        'batch_size': 64,
        'lr': 0.1,
        'weight_decay': 0,
        'n_epochs': 400,
        'n_groups_per_batch': 3,
        'groupby_fields': ['group'],
    },
 ```
 
 ```python
 'rot_simple': {
        'split_scheme': 'official',
        'model': 'logistic_regression',
        'loss_function': 'cross_entropy',
        'optimizer': 'SGD',
        'model_kwargs': {'in_features': 2},
        'algo_log_metric': 'accuracy',
        'val_metric': 'loss_all',
        'val_metric_decreasing': False,
        'batch_size': 64,
        'lr': 0.1,
        'weight_decay': 0,
        'n_epochs': 400,
        'n_groups_per_batch': 3,
        'groupby_fields': ['group'],
    },
 ```
 
 ```python 
 'spu_2feature': {
        'split_scheme': 'official',
        'model': 'logistic_regression',
        'loss_function': 'cross_entropy',
        'optimizer': 'SGD',
        'model_kwargs': {'in_features': 3},
        'algo_log_metric': 'accuracy',
        'val_metric': 'loss_all',
        'val_metric_decreasing': False,
        'batch_size': 64,
        'lr': 0.1,
        'weight_decay': 0,
        'n_epochs': 400,
        'n_groups_per_batch': 3,
        'groupby_fields': ['group'],
    },
```
 
 6. Now that the code is complete, the following commands can be entered into the terminal to obtain results. 
 ```
  python run_expt.py --dataset DATASET --algorithm ALG --root_dir data --progress_bar --log_dir logs/"DATASET"/ALG/run_6_seed_"SEED" --seed SEED  --n_epochs 200
 ```

DATASET = noisy_2feature, rot_simple, spu_2feature <br>
ALG = CG, groupDRO <br>
SEED = 10, 11, 12, 13, 14, 15

 
 ## Extensions
 There have been several extensions to this code and paper introduced and implemented by us. The following steps will explain how we have done that and which commands to use to reproduce these resutls.
 
 ### Multiple groups
1. Repeat the steps from the qualitative analysis but now for the following file-names:
```python
'rot_5group',
'noisy_5group',
'spu_5group'
 ```
 
 2. The parameters added in step 5 of the qualitative analysis are now the following:
 ```python
 'rot_5group': {
        'split_scheme': 'official',
        'model': 'logistic_regression',
        'loss_function': 'cross_entropy',
        'optimizer': 'SGD',
        'model_kwargs': {'in_features': 2},
        'algo_log_metric': 'accuracy',
        'val_metric': 'loss_all',
        'val_metric_decreasing': False,
        'batch_size': 64,
        'lr': 0.1,
        'weight_decay': 0,
        'n_epochs': 400,
        'n_groups_per_batch': 5,
        'groupby_fields': ['group'],
    },
 ```
 
 ```python
 'noisy_5group': {
        'split_scheme': 'official',
        'model': 'logistic_regression',
        'loss_function': 'cross_entropy',
        'optimizer': 'SGD',
        'model_kwargs': {'in_features': 2},
        'algo_log_metric': 'accuracy',
        'val_metric': 'loss_all',
        'val_metric_decreasing': False,
        'batch_size': 64,
        'lr': 0.1,
        'weight_decay': 0,
        'n_epochs': 400,
        'n_groups_per_batch': 5,
        'groupby_fields': ['group'],
    },
 ```
 
 ```python
 'spu_5group': {
        'split_scheme': 'official',
        'model': 'logistic_regression',
        'loss_function': 'cross_entropy',
        'optimizer': 'SGD',
        'model_kwargs': {'in_features': 3},
        'algo_log_metric': 'accuracy',
        'val_metric': 'loss_all',
        'val_metric_decreasing': False,
        'batch_size': 64,
        'lr': 0.1,
        'weight_decay': 0,
        'n_epochs': 400,
        'n_groups_per_batch': 5,
        'groupby_fields': ['group'],
    },
 ```

