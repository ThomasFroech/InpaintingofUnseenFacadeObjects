# Inpainting of Unseen Facade Objects
<p align="center">
<img src="https://github.com/user-attachments/assets/f1f02d4f-c9ba-4387-a09b-833e048c2114" alt="Figure_1_version_4" width="400"/>
</p>

This repository contains the corresponding implementation and supplementary material for the paper _FacaDiffy: Inpainting unseen facade parts using diffusion models_.

**Please note that the code in this repository is purely experimental and will not be maintained**



![Figure_4_with_Numbers drawio(3)](https://github.com/user-attachments/assets/f80e34dd-3191-4f6a-ac70-ae9a266f819d)

## :arrow_forward: How to run?
Since the stable Diffusion Image Inpainting model got taken down from HF, you need to alternatively use the Models provided via Model Scope

1. Download the Stable Diffucion Image Inpainting Model from [ModelScope](https://modelscope.cn/models/AI-ModelScope/stable-diffusion-inpainting/files). 
  
2. Download the FacaDiffy checkpoint from this repository

3. Adjust the paths in Inpainting.ipynb or your own script

##  Reference

If you use this repo please consider linking it or citing the [paper](https://arxiv.org/abs/2502.14940)
```plain
@article{facadiffy,
      title={FacaDiffy: Inpainting unseen facade parts using diffusion models}, 
      author={Thomas Fröch and Olaf Wysocki and Yan Xia and Junyu Xie and Benedikt Schwab and Daniel Cremers and Thomas Heinrich Kolbe},
      year={2025},
journal = {Accepted for ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences (ISPRS Geospatial Week)},
}
```
