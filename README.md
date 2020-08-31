# AtlantaNet
Pytorch implementation of the ECCV 2020 paper: AtlantaNet: Inferring the 3D Indoor Layout from a Single 360 Image beyond the Manhattan World Assumption

![](assets/teaser.jpg)
Images obtained with resnet101_atlantalayout.pth (see below).

This repo is a **python** implementation where you can try:
- **Inference on panoramic images** to get general shaped room layout (Manhattan or Atlanta World) as .json output (3D model viewing included).
We additionally provide numerical evaluation compared to ground truth (see repository).

3D viewer and evaluation metrics are based on the code provided by HorizonNet(https://github.com/sunset1995/HorizonNet) .

**News, 2020-08-31** - Data splitting and matching pre-trained models updated. See provided informations.

**Method Pipeline overview**:
![](assets/overview.jpg)

## Updates
* 2020-08-31: IMPORTANT UPDATE: fixing several issues
	- OpenCV polygonal approximation for .json export
	- Best valid epochs provided now matching with final layout
	- Data splitting and annotation updated
* 2020-08-26: MatterportLayout pre-trained model trained using the original dataset splitting (including non Atlanta World scenes).
* 2020-08-25: Adopted MatterportLayout data splitting added and related information updated.


## Requirements
- Python 3.6
- pytorch>=1.0.1
- numpy
- scipy
- sklearn
- Pillow
- tqdm
- opencv-python>=4.1.1 (for layout simplification and output export)
- open3d>=0.8 (for layout 3D viewer)

## Download Pretrained Models
To be copied in your local ./ckpt directory.
- [resnet50_matterportlayout_origin.pth](https://vicserver.crs4.it/atlantanet/resnet50_matterportlayout_origin.pth)
    - Trained with ResNet50 using MatterportLayout original splitting.  
	    - NB: Includes scenes that do not respect the Atlanta World and Indoor World (single ceiling, vertical walls) hypothesis.
- [resnet50_matterportlayout_iw.pth](https://vicserver.crs4.it/atlantanet/resnet50_matterportlayout_iw.pth)
    - Trained with ResNet50 on MatterportLayout cleaned dataset (data/splitting). 
	    - NB: This fitered data, is adopted in the paper combined to the AtlantaLayout dataset (see paper Sec.5.1 for results). Compared to the original version of MatterportLayout is filtered by scenes that do not respect the Indoor World (single ceiling, Manhattan walls) or Atlanta World (single ceiling, vertical walls) hypothesis.
		Since several annotations have been refined to better adhere to the ceiling profile and the Atlanta/Indoor assumptions, ee both provide splitting files and updated annotations.
- [resnet50_atlantalayout.pth](https://vicserver.crs4.it/atlantanet/resnet50_atlantalayout.pth)
    - Trained with ResNet50 on MatterportLayout cleaned dataset and finetuned on the AtlantaLayout training set.
- [resnet101_atlantalayout.pth](https://vicserver.crs4.it/atlantanet/resnet101_atlantalayout.pth)
    - Trained with ResNet101 on MatterportLayout cleaned dataset and finetuned on the AtlantaLayout training set.

It should be noted that results are obtained converting PanoAnnotator (https://github.com/SunDaDenny/PanoAnnotator) annotations, which are general Manhattan World scenes, to Indoor World model scenes (assumption adopted by LayoutNet, DulaNet and HorizonNet - see https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.14021 for details about such priors). 
Due to this conversion and opencv polygonal approximation, numerical performances can slighly differ from those presented in the paper.

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

	
## Citation
Please cite our paper for any purpose of usage.
```
@InProceedings{Pintore:2020:AI3,
    author = {Giovanni Pintore and Marco Agus and Enrico Gobbetti},
    title = {{AtlantaNet}: Inferring the {3D} Indoor Layout from a Single 360 Image beyond the {Manhattan} World Assumption},
    booktitle = {Proc. ECCV},
    month = {August},
    year = {2020},
    url = {http://vic.crs4.it/vic/cgi-bin/bib-page.cgi?id='Pintore:2020:AI3'},
}
```




