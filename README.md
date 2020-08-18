# AtlantaNet
Pytorch implementation of the ECCV 2020 paper: AtlantaNet: Inferring the 3D Indoor Layout from a Single 360 Image beyond the Manhattan World Assumption

![](assets/teaser.jpg)

This repo is a **python** implementation where you can try:
- **Inference on your images** to get general shaped room layout (Manhattan or Atlanta World) as .json output. 
Additionally we provide:
- **3D output model visualization**
- **Numerical evaluation compared to ground truth**
based on HorizonNet implementation (https://github.com/sunset1995/HorizonNet) to facilitate comparisons.

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
- opencv-python>=3.1 (for output layout simplification)
- open3d>=0.7 (for layout 3D viewer)

## Download Pretrained Models
To be copied in your local ./ckpt directory.
- [resnet50_matterportlayout.pth](https://vicserver.crs4.it/atlantanet/resnet50_matterportlayout.pth)
    - Trained with ResNet50 on MatterportLayout cleaned dataset. 
	    - NB: several images have been removed by the authors due to incorrect annotations or layouts that are not compatible with the original assumptions (e.g. horizontal ceilings). See the original dataset link for details (https://github.com/ericsujw/Matterport3DLayoutAnnotation).
        - Adopted data splitting provided in the data/splitting folder.
- [resnet50_atlantalayout.pth](https://vicserver.crs4.it/atlantanet/resnet50_atlantalayout.pth)
    - Trained with ResNet50 on MatterportLayout cleaned dataset and finetuned on the Atlantalayout training set.
- [resnet101_atlantalayout.pth](https://vicserver.crs4.it/atlantanet/resnet101_atlantalayout.pth)
    - Trained with ResNet101 on MatterportLayout cleaned dataset and finetuned on the Atlantalayout training set.

## Download Dataset
We follow the same notation (.png image with .txt associated) proposed by HorizonNet (https://github.com/sunset1995/HorizonNet).
Instruction to download and prepare PanoContext/Stanford2D3D, MatterportLayout, Structured3D datasets are provided by HorizonNet (https://github.com/sunset1995/HorizonNet) and MatterportLayout(https://github.com/ericsujw/Matterport3DLayoutAnnotation).

- AtlantaLayout Dataset
        - COMING SOON
	
## Inference on your images	



