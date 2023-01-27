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

1. Similar to the case of C-MNIST, references were added to [get_dataset.py](wilds/get_dataset.py).
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
 
 
 ## Extensions
 There have been several extensions to this code and paper introduced and implemented by us. The following steps will explain how we have done that and which commands to use to reproduce these resutls.
 
 ### Multiple groups
 1. A reference to the dataset with 5 rotational groups was added to [get_dataset.py](wilds/get_dataset.py)
 ```python
 elif dataset == 'rot_5group':
        from wilds.datasets.rot_5group_dataset import Rot5GroupDataset
        return Rot5GroupDataset(**dataset_kwargs)
  ```
 
 ### Asymmetric label distribution
 
 ### Upweighted CGD
