# MML Studio 11: Graph Neural Networks for Molecular Property Prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/digitalmoleculardesign/mml_studio_11/blob/main/GNN_Molecular_Property_Prediction_MODERN.ipynb)

## Course: 06-731 Molecular Machine Learning
**Carnegie Mellon University - Gomes Group**

---

## Overview

This studio explores **Graph Neural Networks (GNNs)** for molecular property prediction, treating molecules as graphs where atoms are nodes and bonds are edges. We implement message-passing neural networks (MPNNs) to predict molecular solubility using modern PyTorch tools.

### Learning Objectives

By completing this studio, you will:

1. **Understand** why graph representations are natural for molecular data
2. **Construct** molecular graphs from SMILES strings with atom and bond features
3. **Implement** message-passing neural networks using PyTorch Geometric
4. **Train** and evaluate GNN models with PyTorch Lightning
5. **Use** production-ready tools like Chemprop for real-world applications
6. **Apply** Bayesian optimization and uncertainty quantification

## Background

Graph Neural Networks have become the state-of-the-art architecture for molecular machine learning tasks. By explicitly encoding the molecular graph structure (atoms and bonds), GNNs can learn chemical representations that outperform traditional molecular descriptors.

### Key Concepts

- **Molecular Graphs**: Atoms as nodes, bonds as edges
- **Message Passing**: Iterative aggregation of neighbor information
- **Graph Pooling**: Converting node features to graph-level predictions
- **ESOL Dataset**: Aqueous solubility prediction benchmark

## Contents

- `GNN_Molecular_Property_Prediction_MODERN.ipynb` - Main tutorial notebook (2024-2025 modern stack)
- `Graph_Networks_on_Molecular_Graphs_MML_2025_studio_11_1.ipynb` - Reference implementation (legacy)
- `CLAUDE.md` - Project conventions and technical guidance
- `requirements.txt` - Python dependencies

## Quick Start

### Option 1: Google Colab (Recommended)

Click the "Open in Colab" badge above. All dependencies install automatically.

**Expected Runtime**: 5-10 minutes on T4 GPU

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/digitalmoleculardesign/mml_studio_11.git
cd mml_studio_11

# Create conda environment (use mml_comp_chem if available)
conda create -n mml_studio_11 python=3.11 -y
conda activate mml_studio_11

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install PyTorch Geometric
conda install pyg -c pyg -y

# Install other dependencies
pip install -r requirements.txt

# Install RDKit via conda
conda install -c conda-forge rdkit -y

# Launch Jupyter
jupyter notebook GNN_Molecular_Property_Prediction_MODERN.ipynb
```

## Notebook Structure

### Section 0: Setup & Environment
- Package installation (PyTorch 2.x, PyG 2.6+, Lightning 2.1)
- GPU detection and configuration
- Reproducibility setup
- Gomes Group visualization styling

### Section 1: Introduction
- Why graphs for molecules?
- Limitations of traditional descriptors
- Visual molecular examples

### Section 2: Molecular Graph Construction
- SMILES to graph conversion
- Atom features (9D): atomic number, degree, charge, hybridization, aromaticity
- Bond features (3D): bond type, stereo, conjugation
- PyG Data objects with COO edge indices

### Section 3: ESOL Solubility Dataset
- Dataset loading and exploration
- Distribution analysis
- Train/validation/test splits (70/10/20)
- Target normalization with StandardScaler

### Section 4: Message-Passing Neural Networks - Theory
- Graph convolution concepts
- Message aggregation mechanisms
- Node update functions
- Graph pooling strategies

### Section 5: MPNN Implementation
- Complete PyTorch Geometric implementation
- AtomEncoder and BondEncoder (OGB-style)
- NNConv message passing layers
- GRU node updates
- PyTorch Lightning module

### Section 6: Training & Evaluation
- DataLoader setup for molecular graphs
- Lightning Trainer with callbacks
- Loss curves and metrics tracking
- Test set evaluation (R², RMSE, MAE)
- Parity plots and residuals analysis

### Section 7: Understanding Learned Representations
- Extracting graph embeddings
- PCA visualization (2D projection)
- Chemical space analysis
- Clustering by molecular properties

### Section 8: Chemprop - Production-Ready GNNs
- Chemprop v2.2.1 overview
- CLI training workflow
- Comparison with custom PyG models
- When to use Chemprop vs custom implementations

### Section 9: Common Pitfalls & Debugging
- GPU detection issues
- CUDA out of memory errors
- Poor model performance
- Overfitting and underfitting
- Troubleshooting guide

### Section 10: Extensions & Advanced Topics
- **Bayesian Optimization**: Hyperparameter tuning with Gaussian Processes
- **Uncertainty Quantification**: Ensembles and MC Dropout
- **Advanced Architectures**: Attention mechanisms, graph transformers
- **Further Reading**: Latest research and resources

## Tech Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| **PyTorch** | 2.x | Deep learning framework |
| **PyTorch Geometric** | 2.6+ | Graph neural networks |
| **PyTorch Lightning** | 2.1+ | Training orchestration |
| **RDKit** | 2024.x | Cheminformatics toolkit |
| **Chemprop** | 2.2.1 | Production GNN framework |
| **W&B** | Latest | Experiment tracking |
| **OGB** | Latest | Molecular encoders |

## Expected Results

After training, you should achieve:
- **Test R² Score**: 0.75 - 0.85
- **Test RMSE**: 0.8 - 1.0
- **Test MAE**: 0.6 - 0.8

Performance can vary based on random seed and hardware.

## Key References

### Papers

1. **Gilmer, J. et al.** "Neural Message Passing for Quantum Chemistry" *ICML* (2017) [arXiv](https://arxiv.org/abs/1704.01212)

2. **Yang, K. et al.** "Analyzing Learned Molecular Representations for Property Prediction" *JCIM* (2019) [DOI](https://doi.org/10.1021/acs.jcim.9b00237)

3. **Wieder, O. et al.** "A compact review of molecular property prediction with graph neural networks" *Drug Discovery Today* (2020) [DOI](https://doi.org/10.1016/j.drudis.2020.01.010)

4. **Heid, E. & Green, W. H.** "Chemprop: A Machine Learning Package for Chemical Property Prediction" *JCIM* (2024) [DOI](https://doi.org/10.1021/acs.jcim.3c01250)

### Tutorials & Documentation

- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/)
- [Chemprop Documentation](https://chemprop.readthedocs.io/)
- [OGB Molecular Encoders](https://github.com/snap-stanford/ogb)

## Dataset Reference

**ESOL (Estimated SOLubility)** - Delaney, J. S. "ESOL: Estimating Aqueous Solubility Directly from Molecular Structure" *J. Chem. Inf. Comput. Sci.* (2004) [DOI](https://doi.org/10.1021/ci034243x)

## Original Source

This notebook synthesizes content from:
- [schwallergroup/ai4chem_course](https://github.com/schwallergroup/ai4chem_course)
- [beangoben/chemistry_ml_colab](https://github.com/beangoben/chemistry_ml_colab)

Modernized and enhanced for the 2024-2025 academic year.

## License

MIT License

---

**Gomes Group** | Carnegie Mellon University | [gomesgroup.github.io](https://gomesgroup.github.io)
