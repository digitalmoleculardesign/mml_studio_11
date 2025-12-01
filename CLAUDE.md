# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational course materials for **Graph Neural Networks (GNNs) in Chemistry** - teaching machine learning for molecular property prediction. Based on the AI4Chem course (schwallergroup/ai4chem_course).

## Tech Stack

- **PyTorch** + **PyTorch Lightning** - Deep learning framework
- **PyTorch Geometric (PyG)** - Graph neural network library
- **RDKit** - Cheminformatics toolkit for molecular processing
- **Chemprop** - Message passing networks for molecules
- **Weights & Biases** - Experiment tracking
- **OGB** - Open Graph Benchmark encoders

## Notebook Learning Path

1. `01_intro_to_dl.ipynb` - PyTorch/Lightning basics, feedforward NN on ESOL dataset
2. `02_graph_nns.ipynb` - Molecular graphs, message passing neural networks (MPNN)
3. `03_gnn_simple_example.ipynb` - Chemprop CLI training workflow
4. `Copy_of_*.ipynb` - Advanced topics (node classification, graph classification, scaling, explainability)

## Key Commands

**Package Installation:**
```bash
pip install pytorch-lightning wandb rdkit ogb deepchem torch torch_geometric chemprop
pip install torch-scatter torch-sparse  # PyG dependencies
```

**Chemprop Training:**
```bash
chemprop_train --data_path data/esol.csv \
               --dataset_type regression \
               --save_dir esol_ckpts \
               --metric rmse \
               --split_sizes 0.7 0.1 0.2 \
               --epochs 60
```

## Architecture Patterns

### PyTorch Lightning Module Structure
All models inherit from `pl.LightningModule` with standard methods: `__init__`, `forward`, `training_step`, `validation_step`, `test_step`, `configure_optimizers`, and dataloader methods.

### Graph Data Format (PyG)
```python
# Edge index in COO format: [2, num_edges]
edge_index = torch.tensor([[src_nodes], [dst_nodes]], dtype=torch.long)

# Data object
data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, y=label)
```

### Molecular Feature Extraction
- Atom features: atomic number, degree, formal charge, hybridization, aromaticity
- Bond features: bond type, stereo configuration, conjugation
- Use `smiles2graph()` from OGB or custom RDKit extraction

### MPNN Architecture
```
Atom Embedding → NNConv (message passing) → GRU (node update) → [repeat 3x] → Global Pooling → MLP → Output
```

## Important Conventions

- **Data normalization**: Always normalize targets before training, denormalize for metrics (use sklearn StandardScaler pattern)
- **Reproducibility**: Set seeds for torch, numpy, random at notebook start
- **Data splits**: 70/10/20 train/val/test
- **Hyperparameters**: batch_size=32, lr=0.001, hidden_size=64-512
- **Checkpointing**: Use `trainer.test(ckpt_path="best")` for final evaluation
- **Metrics**: Report RMSE for regression tasks

## Chemistry-Specific Notes

- SMILES strings are the primary molecular input format
- RDKit handles implicit hydrogens by default
- Aromatic vs aliphatic bond types matter for feature extraction
- Number of message passing iterations = graph receptive field (hops)
