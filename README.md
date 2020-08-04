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
- opencv-python>=3.1 (for pre-processing)
- open3d>=0.7 (for layout 3D viewer)

## Download Pretrained Models
To be copied in your local ./ckpt directory.
- [matterportlayout.pth](https://vicserver.crs4.it/atlantanet/matterportlayout.pth)
    - Trained with ResNet50 on MatterportLayout cleaned dataset. 
	    - NB: many images have been corrected or removed by the authors because they have incorrect automatic annotations or layouts that are not compatible with the original assumptions (e.g. ceilings with different heights). See orignal dataset link for details
        - Adopted splitting provided in the repository.


## Download Dataset
Please relative instruction from PanoContext/Stanford2D3D, MatterportLayout, Structured3D datasets providers.
Useful informations and details about how to prepare such data are the HorizonNet and MatterportLayout pages.

- Structured3D Dataset
    - Please contact [Structured3D](https://structured3d-dataset.org/) to get the datas.
    - Following [this](https://github.com/sunset1995/HorizonNet/blob/master/README_ST3D.md#dataset-preparation) to prepare training/validation/testing for HorizonNet.




