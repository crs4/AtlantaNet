# AtlantaNet
Pytorch implementation of the ECCV 2020 paper: AtlantaNet: Inferring the 3D Indoor Layout from a Single 360 Image beyond the Manhattan World Assumption

![](assets/teaser.jpg)
Images obtained with resnet101_atlantalayout.pth (see below).

This repo is a **python** implementation where you can try:
- **Inference on panoramic images** to get general shaped room layout (Manhattan or Atlanta World) as .json output (3D model viewing included).
We additionally provide numerical evaluation compared to ground truth (see repository).

3D viewer and evaluation metrics are based on the code provided by HorizonNet(https://github.com/sunset1995/HorizonNet) to simplify comparisons.

**Method Pipeline overview**:
![](assets/overview.jpg)

## Updates

* 2020-08-25: Adopted MatterportLayout data splitting added and related information updated.


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
	    - NB: This data splitting, adopted in the paper, compared to the original version of matterportlayout is filtered by scenes that do not respect the Indoor World (single ceiling, Manhattan walls) or Atlanta World (single ceiling, vertical walls) hypothesis. 
- [resnet50_atlantalayout.pth](https://vicserver.crs4.it/atlantanet/resnet50_atlantalayout.pth)
    - Trained with ResNet50 on MatterportLayout cleaned dataset and finetuned on the Atlantalayout training set.
- [resnet101_atlantalayout.pth](https://vicserver.crs4.it/atlantanet/resnet101_atlantalayout.pth)
    - Trained with ResNet101 on MatterportLayout cleaned dataset and finetuned on the Atlantalayout training set.

## Download Dataset
We follow the same notation (.png image with .txt associated) proposed by HorizonNet (https://github.com/sunset1995/HorizonNet).
Instruction to download and prepare PanoContext/Stanford2D3D, MatterportLayout, Structured3D datasets are provided by HorizonNet (https://github.com/sunset1995/HorizonNet) and MatterportLayout(https://github.com/ericsujw/Matterport3DLayoutAnnotation).

- AtlantaLayout Dataset
        - Download from here: https://vicserver.crs4.it/atlantanet/atlantalayout.zip
	
## Inference on equirectagular images	
Here an example of inferring using the pre-trained model on MatterportLayout finetuned on AtlantaLayout:
```
python inference_atlanta_net.py --pth ckpt/resnet50_atlantalayout.pth --img data/atlantalayout/test/img/2t7WUuJeko7_c2e11b94c07a4d6c85cc60286f586a02_equi.png
```    
    - `--pth` path to the trained model.
    - `--img` path to the input equirectangular image.
    - `--output_dir` path to the directory to dump .json results (Optional: default = results/).
    - `--visualize` optional for visualizing textured 3D model (Optional: default = True).
	```




