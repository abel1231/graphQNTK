# GraphQNTK

This repository contains code implementation for the paper **GraphQNTK: Quantum Neural Tangent Kernel for Graph Data, *NeurIPS*, 2022**. Our implementation is based on the code[(url)](https://github.com/KangchengHou/gntk) and we thank the authors for sharing.

## Packages requirements

We use the following version of the packages to run the code, and a later version should also work properly.

```
networkx 2.2.1
numpy 1.19.5
scikit-learn 0.23.2
scipy 1.5.3
```

## Datasets download

The datasets are available at this [site](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets). Download the compressed file and extract it into ```./dataset``` under the root directory.


## Train the model

To obtain the kernel matrix of the enhanced graph neural tangent kernel
```
python train.py --dataset MUTAG --num_mlp_layers 2 --num_layers 4 --scale uniform --out_dir out
```
The kernel matrix would be saved in thr file ```./out/{dataset folder}/gram.npy```. To make predictions based on the pre-defined kernel
```
python search.py --data_dir ./out/{dataset folder} --dataset MUTAG
```
The results would be found in ```./out/{dataset folder}/grid_search.csv```.

In order to simplify the estimation of the prediction results of different hyperparameters (such as different layers) under the same dataset, we provide a python script ```conclude.py``` for convenience. To output the results in one ```.csv``` file, run the code
```
python conclude.py --data_dir out --dataset MUTAG
```
The results would be saved in ```./out/conclude-{dataset}.csv```.









