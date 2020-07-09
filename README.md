# AtlantaNet
Pytorch implementation of the ECCV 2020 paper: AtlantaNet: Inferring the 3D Indoor Layout from a Single 360 Image beyond the Manhattan World Assumption

![](assets/teaser.jpg)

This repo is a **python** implementation that you can:
- **Inference on your images** to get cuboid or general shaped room layout
- **3D layout viewer**

**Method Pipeline overview**:
![](assets/overview.jpg)

## Requirements
- Python 3
- pytorch>=1.0.0
- numpy
- scipy
- sklearn
- Pillow
- tqdm
- tensorboardX
- opencv-python>=3.1 (for pre-processing)
- open3d>=0.7 (for layout 3D viewer)
