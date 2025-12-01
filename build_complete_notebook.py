#!/usr/bin/env python3
"""
Script to build the complete GNN Molecular Property Prediction notebook.
This integrates content from all source notebooks into a comprehensive modern tutorial.

Author: Gomes Research Group, CMU
Date: December 2024
"""

import json
import sys
from pathlib import Path

def create_markdown_cell(content):
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": content.split('\n')
    }

def create_code_cell(content):
    """Create a code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": content.split('\n')
    }

def build_section_1():
    """Section 1: Introduction - Why Graphs for Molecules?"""
    cells = []

    # Section header
    cells.append(create_markdown_cell("""<a id='introduction'></a>
# 1. Introduction: Why Graphs for Molecules?

**⏱️ Expected time:** 10 minutes

## The Challenge of Molecular Representation

Molecules are complex 3D structures with atoms connected by chemical bonds. How should we represent them for machine learning?

### Traditional Approaches

1. **SMILES Strings** → Treat as text sequences
2. **Molecular Descriptors** → Hand-crafted features (MW, logP, TPSA, etc.)
3. **Fingerprints** → Binary vectors (Morgan, MACCS, etc.)

### The Graph Representation Advantage

A **graph** is a natural way to represent molecules:
- **Nodes (Vertices)** = Atoms
- **Edges (Links)** = Chemical bonds
- **Node features** = Atom properties (element, charge, hybridization)
- **Edge features** = Bond properties (type, stereochemistry)

<div align="center">
<img src="https://github.com/beangoben/chemistry_ml_colab/blob/master/images/Chloroquine-2D-molecular-graph.png?raw=1" width="500"/>
</div>

### Why This Matters

> **Inductive Bias**: Graphs encode the assumption that molecular properties are determined by:
> 1. The types of atoms present
> 2. How atoms are connected
> 3. Local chemical environments (neighborhoods)

This is exactly how chemists think about molecules!"""))

    # Visualization example
    cells.append(create_code_cell("""print_section_header("Visualizing Molecular Graphs", "🔬")

# Example molecules
molecules = {
    'Methane': 'C',
    'Ethanol': 'CCO',
    'Benzene': 'c1ccccc1',
    'Aspirin': 'CC(=O)Oc1ccccc1C(=O)O'
}

# Visualize molecules
mols = [Chem.MolFromSmiles(smi) for smi in molecules.values()]
img = Draw.MolsToGridImage(
    mols,
    molsPerRow=2,
    subImgSize=(300, 300),
    legends=list(molecules.keys())
)
display(img)

print("\\n✓ Each molecule is represented as a graph:")
print("  - Nodes: atoms (C, O, H, etc.)")
print("  - Edges: bonds (single, double, aromatic)")"""))

    # Graph theory basics
    cells.append(create_markdown_cell("""## Graph Theory Basics

A graph $G = (V, E)$ consists of:

- **Vertices (Nodes)**: $V = \\{v_1, v_2, ..., v_n\\}$
- **Edges**: $E \\subseteq \\{(i,j) | i,j \\in V, i \\neq j\\}$

For molecules, we typically use **undirected graphs** (bonds work both ways).

### Graph Properties

| Property | Molecular Interpretation |
|----------|--------------------------|
| **Node degree** | Number of bonds an atom forms |
| **Path length** | Shortest distance between atoms |
| **Cycles** | Rings in molecules |
| **Connectivity** | Whether molecule is fragmented |

### Adjacency Matrix

An $n \\times n$ matrix $A$ where $A_{ij} = 1$ if atoms $i$ and $j$ are bonded:

$$
A = \\begin{bmatrix}
0 & 1 & 0 & 0 \\\\
1 & 0 & 1 & 1 \\\\
0 & 1 & 0 & 1 \\\\
0 & 1 & 1 & 0
\\end{bmatrix}
$$

**Note**: For GNNs, we typically use the more efficient **edge index** format instead."""))

    # Interactive example
    cells.append(create_code_cell("""# Interactive example: Build a graph for caffeine
print_section_header("Example: Caffeine Molecular Graph", "☕")

caffeine_smiles = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
mol = Chem.MolFromSmiles(caffeine_smiles)

print(f"Molecule: Caffeine")
print(f"SMILES: {caffeine_smiles}")
print(f"Molecular Formula: {Chem.rdMolDescriptors.CalcMolFormula(mol)}")
print(f"\\nGraph Properties:")
print(f"  Number of atoms (nodes): {mol.GetNumAtoms()}")
print(f"  Number of bonds (edges): {mol.GetNumBonds()}")
print(f"  Number of rings: {Chem.rdMolDescriptors.CalcNumRings(mol)}")

# Display molecule
display(mol)

# Calculate and display adjacency matrix
from rdkit.Chem import GetAdjacencyMatrix
adj_matrix = GetAdjacencyMatrix(mol)

print(f"\\nAdjacency Matrix Shape: {adj_matrix.shape}")
print("First 5x5 block:")
print(adj_matrix[:5, :5])"""))

    # Key insights
    cells.append(create_markdown_cell("""## Key Insights

1. **Permutation Invariance**: Molecular properties don't depend on the order we list atoms
   - Need models that respect this symmetry

2. **Variable Size**: Molecules have different numbers of atoms
   - Need models that handle variable-size inputs

3. **Local + Global**: Properties emerge from both local (functional groups) and global (overall structure) features
   - Need models that aggregate information at multiple scales

**Graph Neural Networks (GNNs)** are designed to address all three challenges!

---"""))

    return cells

def build_section_2():
    """Section 2: Molecular Graph Construction"""
    cells = []

    cells.append(create_markdown_cell("""<a id='graph-construction'></a>
# 2. Molecular Graph Construction

**⏱️ Expected time:** 15 minutes

Now we'll learn how to convert SMILES strings into graph data structures that can be processed by GNNs.

## 2.1 Atom Features

What information should we extract from each atom?

| Feature | Description | Example Values |
|---------|-------------|----------------|
| **Atomic Number** | Element identity | 1 (H), 6 (C), 7 (N), 8 (O) |
| **Degree** | Number of bonded neighbors | 0, 1, 2, 3, 4 |
| **Formal Charge** | Ionic charge | -1, 0, +1 |
| **Hybridization** | Orbital type | sp, sp², sp³ |
| **Aromaticity** | In aromatic ring? | True/False |
| **In Ring** | Part of cycle? | True/False |

We'll use the **OGB (Open Graph Benchmark)** featurization scheme, which is well-tested and comprehensive."""))

    cells.append(create_code_cell("""# Define allowable atom features (from OGB)
ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)) + ['misc'],  # Elements 1-118
    'degree': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'formal_charge': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'chiral_tag': ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER'],
    'num_Hs': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'hybridization': ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True]
}

def safe_index(lst, element):
    """
    Return index of element in list. If not present, return last index (misc).
    """
    try:
        return lst.index(element)
    except ValueError:
        return len(lst) - 1

def atom_to_feature_vector(atom):
    """
    Convert RDKit atom object to feature vector (list of indices).

    Args:
        atom: RDKit Atom object

    Returns:
        list: Feature vector with 9 integer indices
    """
    features = [
        safe_index(ATOM_FEATURES['atomic_num'], atom.GetAtomicNum()),
        safe_index(ATOM_FEATURES['degree'], atom.GetTotalDegree()),
        safe_index(ATOM_FEATURES['formal_charge'], atom.GetFormalCharge()),
        ATOM_FEATURES['chiral_tag'].index(str(atom.GetChiralTag())),
        safe_index(ATOM_FEATURES['num_Hs'], atom.GetTotalNumHs()),
        safe_index(ATOM_FEATURES['hybridization'], str(atom.GetHybridization())),
        ATOM_FEATURES['is_aromatic'].index(atom.GetIsAromatic()),
        ATOM_FEATURES['is_in_ring'].index(atom.IsInRing()),
    ]
    return features

# Example: Extract features from ethanol (CCO)
mol = Chem.MolFromSmiles('CCO')
print("Ethanol (CCO) Atom Features:")
print("-" * 50)
for i, atom in enumerate(mol.GetAtoms()):
    features = atom_to_feature_vector(atom)
    print(f"Atom {i} ({atom.GetSymbol()}):")
    print(f"  Atomic number: {atom.GetAtomicNum()}")
    print(f"  Degree: {atom.GetTotalDegree()}")
    print(f"  Hybridization: {atom.GetHybridization()}")
    print(f"  Aromatic: {atom.GetIsAromatic()}")
    print(f"  Feature vector: {features}")
    print()

print(f"Total feature dimension per atom: {len(features)}")"""))

    cells.append(create_markdown_cell("""## 2.2 Bond Features

Bond features capture the nature of chemical connections:

| Feature | Description | Example Values |
|---------|-------------|----------------|
| **Bond Type** | Single, double, triple, aromatic | SINGLE, DOUBLE, TRIPLE, AROMATIC |
| **Stereochemistry** | E/Z, cis/trans configuration | STEREONONE, STEREOE, STEREOZ |
| **Conjugation** | Part of conjugated system? | True/False |"""))

    cells.append(create_code_cell("""# Define allowable bond features
BOND_FEATURES = {
    'bond_type': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC', 'misc'],
    'stereo': ['STEREONONE', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS', 'STEREOANY'],
    'is_conjugated': [False, True],
}

def bond_to_feature_vector(bond):
    """
    Convert RDKit bond object to feature vector.

    Args:
        bond: RDKit Bond object

    Returns:
        list: Feature vector with 3 integer indices
    """
    features = [
        safe_index(BOND_FEATURES['bond_type'], str(bond.GetBondType())),
        BOND_FEATURES['stereo'].index(str(bond.GetStereo())),
        BOND_FEATURES['is_conjugated'].index(bond.GetIsConjugated()),
    ]
    return features

# Example: Extract bond features from ethanol
mol = Chem.MolFromSmiles('CCO')
print("Ethanol (CCO) Bond Features:")
print("-" * 50)
for i, bond in enumerate(mol.GetBonds()):
    features = bond_to_feature_vector(bond)
    begin_atom = bond.GetBeginAtom().GetSymbol()
    end_atom = bond.GetEndAtom().GetSymbol()
    print(f"Bond {i} ({begin_atom}-{end_atom}):")
    print(f"  Type: {bond.GetBondType()}")
    print(f"  Conjugated: {bond.GetIsConjugated()}")
    print(f"  Feature vector: {features}")
    print()

print(f"Total feature dimension per bond: {len(features)}")"""))

    cells.append(create_markdown_cell("""## 2.3 Edge Index Representation

Instead of an adjacency matrix, we use **edge index** format:

$$
\\text{edge\\_index} = \\begin{bmatrix}
\\text{source nodes} \\\\
\\text{target nodes}
\\end{bmatrix} \\in \\mathbb{Z}^{2 \\times |E|}
$$

**Example**: For a molecule with bond between atom 0 and atom 1:

```python
edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # Undirected
```

**Why this format?**
- Memory efficient for sparse graphs (most molecules are sparse)
- Fast operations in PyTorch Geometric
- Easy to add/remove edges"""))

    cells.append(create_code_cell("""# Build edge index for a molecule
def molecule_to_graph_data(smiles):
    """
    Convert SMILES to PyTorch Geometric Data object.

    Args:
        smiles: SMILES string

    Returns:
        Data: PyG Data object with x, edge_index, edge_attr
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(atom_to_feature_vector(atom))
    x = torch.tensor(atom_features, dtype=torch.long)

    # Edge features and edge index
    edge_indices = []
    edge_features = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_feat = bond_to_feature_vector(bond)

        # Add both directions (undirected graph)
        edge_indices.append([i, j])
        edge_features.append(bond_feat)
        edge_indices.append([j, i])
        edge_features.append(bond_feat)

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.long)

    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

# Test on example molecules
examples = [
    ('Methane', 'C'),
    ('Ethanol', 'CCO'),
    ('Benzene', 'c1ccccc1'),
    ('Aspirin', 'CC(=O)Oc1ccccc1C(=O)O')
]

print("Molecular Graph Statistics:")
print("=" * 70)
for name, smiles in examples:
    graph = molecule_to_graph_data(smiles)
    print(f"{name:12s} ({smiles:25s})")
    print(f"  Nodes: {graph.num_nodes:3d}  |  Edges: {graph.num_edges:3d}  |  "
          f"Node feat: {graph.x.shape}  |  Edge feat: {graph.edge_attr.shape}")
print("=" * 70)"""))

    cells.append(create_markdown_cell("""## 2.4 PyTorch Geometric Data Objects

The `Data` class in PyG wraps everything together:

```python
data = Data(
    x=node_features,          # [num_nodes, num_node_features]
    edge_index=edge_index,    # [2, num_edges]
    edge_attr=edge_features,  # [num_edges, num_edge_features]
    y=target_value           # [1] or [num_tasks]
)
```

**Key attributes**:
- `data.x`: Node feature matrix
- `data.edge_index`: Graph connectivity (COO format)
- `data.edge_attr`: Edge feature matrix
- `data.y`: Target label/property
- `data.batch`: Batch assignment vector (for minibatching)

---"""))

    return cells

# Continue with more sections...
def build_notebook():
    """Build the complete notebook."""
    print("Building complete GNN Molecular Property Prediction notebook...")

    # Load existing notebook
    notebook_path = Path('/Users/passos/Downloads/studio_11/GNN_Molecular_Property_Prediction_MODERN.ipynb')
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)

    # Keep Section 0 (setup) - it's already there
    # Add new sections
    new_cells = []
    new_cells.extend(build_section_1())
    new_cells.extend(build_section_2())
    # More sections will be added...

    # Add to notebook
    notebook['cells'].extend(new_cells)

    # Save
    output_path = Path('/Users/passos/Downloads/studio_11/GNN_Molecular_Property_Prediction_COMPLETE.ipynb')
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=2)

    print(f"✓ Notebook saved to: {output_path}")
    print(f"✓ Total cells: {len(notebook['cells'])}")

    return output_path

if __name__ == '__main__':
    build_notebook()
