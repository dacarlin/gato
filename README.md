# Protein design with graph attention networks (GATs)

This repository explores the use of graph attention networks (GATs) to solve the problem of designing protein sequences conditioned on structures, including proteins, nucleic acids, metal ions, and small molecule ligands. I've been calling it "Gato" (a play on "GAT") so that's what the repo's called. 

```
protein backbone coords --> encode as graph --> GAT model --> protein sequence 
```

It's a generative model, you can you sample new sequences with temperature and do other kinds of generation conditioned on structures. 


## Background 

We compare some of the best methods for graph based protein design currently available and see if we can develop some excellent evals for assessing how and why the models perform differently, with an eye towards improving them. 

The main sources of inspiration are: 

- [Generative Models for Graph-Based Protein Design](https://papers.nips.cc/paper/9711-generative-models-for-graph-based-protein-design) by John Ingraham, Vikas Garg, Regina Barzilay and Tommi Jaakkola, NeurIPS 2019. This is the first paper that frames the problem and provides an implementation, which is used in part by the following two papers. 
- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) by Justas Dauparas, 2021 
- [ESM-IF](https://github.com/facebookresearch/esm) by Alex Rives and coworkers, 2021  

From the original implementation and idea by John Ingraham: our goal is to create a model that "'designs' protein sequences for target 3D structures via a graph-conditioned, autoregressive language model". 

However, we take a slightly different approach here. For this work, we build a clean, simple GAT implementation in PyG with education in mind, without reusing any previous code. 


## Goals for this repository 

- [x] Present a simple and understandable implementation of state of the art algorithms for protein design with graph attention networks 
    - [x] Create fast and flexible structure data loaders from PDB for PyTorch Geometric 
    - [x] Implement featurization from Ingraham
    - [x] Implement featurization scheme from ProteinMPNN
- [ ] Perform head to head experiments with existing models 
    - [ ] Train GAT models over hyperparameter grid 
        - [ ] Grid: hidden=64,128,256 layers=2,3,4
    - [ ] Train Ingraham model with coarse feature set 
    - [ ] Train Ingraham model with full feature set 
    - [ ] Train ProteinMPNN model with settings from paper 
- [ ] Analysis of model performance and potential improvements 
    - [ ] Perform analysis of the model attention mechanism 
    - [ ] Devise evals that probe the ability of models under different conditions 


## Code overview 

### Protein design GAT implementation 

GAT implementation: 

- `data.py`. Preprocess data for GAT models 
- `model.py`. Modules and functions that implement the model 
- `train.py`. Implements a GAT model and the training loop

To use your own dataset 

```
python data.py --dataset path/to/pdb/files
python train.py 
```

### Benchmarks 

For the benchmarks, we train models using the code from Ingraham and ProteinMPNN. In addition, we also extract the feature representations from these models as part the feature analysis:

- `prepare_cath_dataset.ipynb`. Create the CATH dataset from raw files. Creates the files `chains.jsonl` and `splits.jsonl`. 
- `compare_features.ipynb`. Compare features from Ingraham and ProteinMPNN.  

| Model | Modification | Params | Test perplexity | Test accuracy |
|-------|--------------|--------|-----------------|---------------|
| Uniform |           |         | 20.0            |           | 
| Natural frequencies |         | | 17.83 |  | 
| Ingraham (h=128,layer=3) | CA only |     | 6.85 |  | 
| ProteinMPNN | CA only | 1.3 M | 6.51 | 41.2 |
| ProteinMPNN | Add N, Ca, C, Cb, O distances | 1.43 M | 5.03 | 49.0 |

ProteinMPNN paper: https://doi.org/10.1101/2022.06.03.494563
Ingraham https://github.com/jingraham/neurips19-graph-protein-design