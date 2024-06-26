# MolEdit: Versatile Molecular Editing via Multimodal and Group-optimized Generative Learning

This is the github repo for the paper ["Versatile Molecular Editing via Multimodal and Group-optimized Generative Learning"](https://doi.org/10.26434/chemrxiv-2023-j2n6l).

MolEdit is a multimodal, group-optimized, physics-informed and controllable generative method for molecular systems, developed by Gao Group, Peking University. It is designed to tackle the challenges of molecular editing, a fundamental problem in functional molecule design, which encompasses the generation, modification and evolution of molecules towards desired properties with specified structural features.

## Multimodal Generation of Molecules 
We propose the multimodal likelihood decomposition to address the "discreteness-continuity duality" inherent in the space of molecules. This approach enables robust generation across multiple molecular modalities, including constituents, graphs, and structures.
![image](https://github.com/issacAzazel/MolEdit/blob/main/figs/generation_example.png)

## Diverse Generation in Chemical and Conformational Space 
MolEdit is capable of generating diversity in both chemical and conformational spaces. It's notable that MolEdit was trained on a dataset containing only one conformation per molecule. The ability to sample multiple conformations is a result of a "zero-shot" generalization.
![image](https://github.com/issacAzazel/MolEdit/blob/main/figs/diversity.png)

## Learning Symmetry-preserving distributions
We develop group-optimized score matching (GOSM) that mathematically incorporates group actions into diffusion models. GOSM can learn distributions that preserve symmetry under any group, including SE(3) and permutation groups commonly found in molecular systems.
![image](https://github.com/issacAzazel/MolEdit/blob/main/figs/GOSM.png)

## Physics-informed Diffusion Models 
To address non-physical and unreasonable structures (e.g., atom clashes, inappropriate bond lengths, incorrect valences) generated by traditional data-driven models, we implement a Boltzmann-Gaussian mixture (BGM) kernel in our diffusion processes. This approach, grounded in statistical mechanics and molecular force fields, significantly mitigates structural issues in generated molecules.
![image](https://github.com/issacAzazel/MolEdit/blob/main/figs/BGM.png)

## Controllable Editing of Generated Molecules
As a compelling feature, MolEdit can operate following various prompts that specify conditional information about molecular constituents, subgraphs as well as substructures, and flexibly generate precisely edited molecular graphs and their corresponding conformers.
![image](https://github.com/issacAzazel/MolEdit/blob/main/figs/editing.png)

## Demo

Explore MolEdit's capabilities with our [moledit.ipynb](./moledit.ipynb) notebook, showcasing a demo of multimodal molecular generation trained on [QM9 dataset](https://www.nature.com/articles/sdata201422).

Running this demo requires: 
* jax==0.4.20, jaxlib==0.4.20
* flax==0.8.3
* rdkit==2023.9.6
* Xponge==1.5.0a6, installed via `pip install git+https://gitee.com/gao_hyp_xyj_admin/xponge.git`
* nglview==3.1.2 (for visualization in notebooks)

## Data Availability

The dataset used in the development of MolEdit can be found in [here](https://zenodo.org/records/11115372).

## Citation
```python
@article{lin2023versatile,
    title={Versatile Molecular Editing via Multimodal and Group-optimized Generative Learning},
    author={Lin, Xiaohan and Xia, Yijie and Huang, Yupeng and Liu, Shuo and Chen, Mengyun and Ni, Ningxi and Wang, Zidong and Gao, Yi Qin and Zhang, Jun},
    year={2023},
}
```

## Contact 
For questions or further information, please contact jzhang@cpl.ac.cn.