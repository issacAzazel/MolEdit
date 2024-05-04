# MolEdit: Versatile Molecular Editing via Multimodal and Group-optimized Generative Learning

This is the github repo for the paper ["Versatile Molecular Editing via Multimodal and Group-optimized Generative Learning"](https://doi.org/10.26434/chemrxiv-2023-j2n6l).

MolEdit is a multimodal, group-optimized, physics-informed and controllable generative method for molecular systems, developed by Gao Group, Peking University. With MolEdit, we aim to solve the challenge of molecular editing, a central concept in functional molecule design, which encompasses the generation, modification and evolution of molecules towards desired properties with specified structural features.

## Multimodal Generation of Molecules 
We propose the multimodal likelihood decomposition 
$$
    \log P(\mathrm{mol})\approx \log P_S(\mathbf{s}_{\mathrm{mol}}) + \log p_{X|S}(\mathbf{x}_{\mathrm{mol}} | \mathbf{s}_{\mathrm{mol}})
$$

## Group-optimized Score Matching 

## Physics-informed Learning 

## Controllable Editing of Generated Molecules

## Demo

Get started with [moledit.ipynb](./moledit.ipynb) notebook, where we implemented a demo of multimodal generaion of molecules, with MolEdit trained on [QM9 dataset](www.nature.com/articles/sdata201422).

Running this demo requires: 
* jax==0.4.20, jaxlib==0.4.20
* flax==0.8.3
* rdkit==2023.9.6
* Xponge==1.5.0a6, installed via `pip install git+https://gitee.com/gao_hyp_xyj_admin/xponge.git`
* nglview==3.1.2 (for visualization in notebooks)

## Citation
```python
@article{lin2023versatile,
    title={Versatile Molecular Editing via Multimodal and Group-optimized Generative Learning},
    author={Lin, Xiaohan and Xia, Yijie and Huang, Yupeng and Liu, Shuo and Chen, Mengyun and Ni, Ningxi and Wang, Zidong and Gao, Yi Qin and Zhang, Jun},
    year={2023},
}
```

## Contact 
If you have any questions, please contact jzhang@cpl.ac.cn.