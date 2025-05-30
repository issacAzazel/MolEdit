# MolEdit: *In-silico* 3D Molecular Editing through Physics-Informed and Peference-Aligned Generative Foundation Models

This is the github repo for the paper "*In-silico* 3D Molecular Editing through Physics-Informed and Peference-Aligned Generative Foundation Models". An early version is preprinted at [doi:10.26434/chemrxiv-2023-j2n6l-v2](doi.org/10.26434/chemrxiv-2023-j2n6l-v2).

MolEdit, developed by the Gao Group at Peking University, is a molecular generative artificial intelligence (GenAI) designed to address the complex challenges of *molecular editing*. This fundamental aspect of functional molecule design involves the generation, modification, and evolution of molecules with specified structural and chemical features. Our goal is to develop molecular GenAIs as powerful as image-based GenAIs, such as image denoising diffusion probabilistic models (DDPMs).

## Scaling 3D Molecular Diffusion Models for Large and Bioactive Molecules
Using our approach, MolEdit scales 3D molecular generation from QM9 (≤9 heavy atoms, small molecules) to ZINC (≤64 heavy atoms, drug-like), and further to QMugs (≤100 heavy atoms, bioactive).
<p align="center"><img src="https://github.com/issacAzazel/MolEdit/blob/main/figs/scaling.png" width="70%"></p>

## Multimodal Generation of Molecules 
We propose an asynchronous multimodal diffusion schedule for molecular DDPMs. This approach makes it possible to rigorously handle molecular symmetries (various point groups), and facilitates robust generation across various molecular modalities, including SMILES, graphs, and 3D structures.<p align="center"><img src="https://github.com/issacAzazel/MolEdit/blob/main/figs/generation_example.png" width="70%"></p>

## Imbue Diffusion Models with Symmetry-Awareness via Group-Optimized Labeling
We develop Group-Optimized Denoising Diffusion (GODD), that explicitly integrates group actions into DDPMs. GODD functions through group-optimized labeling, which serves as a plug-and-play and non-invasive modification to DDPM training objectives. GODD enhances the model's awareness of the diverse molecular symmetries embedded in various molecular point groups.<p align="center"><img src="https://github.com/issacAzazel/MolEdit/blob/main/figs/symmetry.png" width="70%"></p> 
<p align="center"><img src="https://github.com/issacAzazel/MolEdit/blob/main/figs/GOSM.png" width="70%"></p> 

## Hallucination Suppression via Physics-alignment
To suppress hallucinations of physically unstable and chemically invalid structures (atom clashes, inappropriate bond lengths, and incorrect valences) generated by molecular GenAIs, we draw inspiration from preference alignment techniques used in mainstream GenAIs. We implement a Boltzmann-Gaussian mixture (BGM) kernel for physics alignment. This approach, grounded in statistical mechanics and molecular force fields, built upon a "Hamiltonian oracle" and significantly improves the stability of structures in the molecules generated.<p align="center">
<img src="https://github.com/issacAzazel/MolEdit/blob/main/figs/BGM.png" width="70%"></p> 

## Versatile and Controllable Design and Editing of Functional Molecules
Inspired by the success of image GenAIs, a foundational molecular GenAI should be able to respond to various specifications of molecular editing. Therefore, we equip MolEdit with flexible controls in context of multiple conditions. For instance, MolEdit is able to control the shape of molecules using the radius of gyration. The model can also be conditioned on (sub)molecular graphs and "inpaint" the missing motifs given predefined chemical contexts like fragments, groups, etc., and/or given geometric restraints like substructures, orientations, etc. This controllable generation strategy also facilitates MolEdit to render diverse high-quality structures from textual molecular representations like SMILES and graphs.

Detailed examples can be found in notebook [moledit_ZINC.ipynb](./moledit_ZINC.ipynb)!

## Exploring MolEdit's capabilities!

Before running notebooks, download and unzip pre-trained MolEdit checkpoints (`checkpoints.zip`) from [here](https://zenodo.org/records/15480816), and move folder `moledit_dataset` and folder `params` into your working directory.

Explore MolEdit's capabilities with:
* [moledit_qm9.ipynb](./moledit_qm9.ipynb) notebook, which contains demos of generation and editing of molecules (less than 9 non-hydrogen atoms), and physics-alignment via MolEdit trained on [QM9 dataset](https://www.nature.com/articles/sdata201422).
* [moledit_QMugs.ipynb](./moledit_QMugs.ipynb) notebook, which contains demos of property-guided sampling via MolEdit trained on [QMugs dataset](https://www.nature.com/articles/s41597-022-01390-7)
* [moledit_ZINC.ipynb](./moledit_ZINC.ipynb) notebook, which contains demos of de-novo design of drug-like molecules, and following applications via MolEdit trained on [ZINC dataset](https://zinc15.docking.org/).
    * High quality structure rendering and diverse conformational sampling for complex molecules ![image](https://github.com/issacAzazel/MolEdit/blob/main/figs/structure_rendering.png)  
    * Structure editing and inpainting <p align="center"><img src="https://github.com/issacAzazel/MolEdit/blob/main/figs/structure_editing.png" width="50%"></p> 
    * Re-design of aromatic systems <p align="center"><img src="https://github.com/issacAzazel/MolEdit/blob/main/figs/aromatic_system.png" width="50%"></p> 
    * Functional-core based design <p align="center"><img src="https://github.com/issacAzazel/MolEdit/blob/main/figs/functional_core.png" width="50%"></p> 
    * Linker design <p align="center"><img src="https://github.com/issacAzazel/MolEdit/blob/main/figs/linker_design.png" width="50%"></p> 
    * Structure building for free energy perturbation![image](https://github.com/issacAzazel/MolEdit/blob/main/figs/FEP.png) 
    * Lead-imprinted binder design for protein pockets![image](https://github.com/issacAzazel/MolEdit/blob/main/figs/lead_imprinting.png) 
    * ...

Running these notebooks requires: 
* python==3.10
* jax==0.4.20, jaxlib==0.4.20
* flax==0.8.3
* e3nn_jax==0.20.6
* rdkit==2023.9.6
* ml_collections
* Xponge==1.5.0a6, installed via `pip install git+https://gitee.com/gao_hyp_xyj_admin/xponge.git`
* nglview==3.1.2 (for visualization in notebooks)

In theory, any environment compatible with the packages mentioned above should run successfully. Our configuration includes Ubuntu 22.04 (GNU/Linux x86_64), NVIDIA A100-SXM4-80GB, CUDA 11.8 and Anaconda 23.7.2. The complete notebook execution takes approximately 0.5 hours.

Details on calculating quantitative metrics, including validity, uniqueness, molecular physical stability, and conformational diversity, can be found in [evaluation](./evaluation). Additionally, we provide useful scripts to evaluate energy/force under [the General AMBER force field](https://ambermd.org/antechamber/gaff.html) (without periodic boundary conditions). We also provide scripts used in quantitative benchmark study (linker design & binder design).

## Training a MolEdit Model

We provide an example [training script](./main_train.py) for implementing MolEdit. To train your model:

### Data Preparation
1. Save each data point as individual `.pkl` files
2. Create a `name_list.pkl` file containing a dictionary with the following structure:
```python
{
    (0, 8): {
        "size": ...,          # [description of this field]
        "name_list": [pkl_file_1, pkl_file_2, ...]
    }, 
    (8, 16): {
        "size": ...,          # [description of this field]
        "name_list": [pkl_file_1, pkl_file_2, ...]
    },
    ...
}
```
- **Dictionary keys**: Tuple ranges indicate atom count intervals (e.g., (0,8) for molecules with 0-8 atoms)
- **Value structure**:
  - `size`: [brief explanation of what this represents]
  - `name_list`: List of paths to your `.pkl` files

### Data File Format
Each `.pkl` file should contain:
```python
{
    'feature': {
        mask_type: {
            'n_atoms': ...,       # Number of atoms
            'atom_feat': ...,     # Atom features [describe format if needed]
            'bond_feat': ...      # Bond features [describe format if needed]
        },
        ...
    },
    'structure': {
        mask_type: {
            'rg': ...,             # Radius of gyration
            'atom_crd': ...,       # Ground truth 3D coordinates
            'perm_transrot_crd': ...,  # Group-optimized noisy structure
            'perm_transrot_label': ... # Group-optimized labels
        },
        ...
    }
}
```
### Alternative Approach
You may alternatively implement your own data pipeline if this format doesn't suit your needs.

## Data Availability

The dataset used in the development of MolEdit can be found in [here](https://zenodo.org/records/15480816).

## Citation
```python
@article{lin2023versatile,
    title={Versatile Molecular Editing via Multimodal and Group-optimized Generative Learning},
    author={Lin, Xiaohan and Xia, Yijie and Huang, Yupeng and Liu, Shuo and Zhang, Jun and Gao, Yi Qin and Zhang, Jun},
    year={2023},
}
```

## Contact 
For questions or further information, please contact jzhang@cpl.ac.cn.