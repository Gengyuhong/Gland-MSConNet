# Gland-MSConNet
**Gland-MSConNet: A semantic segmentation Model for Cotton Leaf Pigment Gland Detection and Phenotypic Quantification**


![Figure_2](https://github.com/user-attachments/assets/3ffdbf2d-8929-4c83-8cbf-1a7746d952fa)


# Getting Started
## Installation
Before running the code, ensure you have all required dependencies installed. You can install them using the following command:

```
pip install -r requirements.txt
```

## Data Preparation
The dataset used in this study is available for download at the following link:

**[Datasets](https://pan.baidu.com/s/1-a8_EDrq0sIySpnlFbvAlg?pwd=luck)** **(extraction code: luck)**


## Training
To train the model, run the following command:

 ```python
 python train.py
```

## Evaluation
To validate the model's performance, run:

 ```python
 python get_miou.py
```
The weights file is located in the model's “logs” folder:
```
last_epoch_weights.h5
```

## Comparison Models
For benchmarking, Gland-MSConNet is compared with the following state-of-the-art segmentation models:


  **[Unet](https://github.com/bubbliiiing/unet-pytorch)**
  
  **[Deeplabv3+](https://github.com/bubbliiiing/deeplabv3-plus-pytorch)**
  
 **[PSPnet](https://github.com/bubbliiiing/pspnet-pytorch)**
  
  **[Transunet](https://github.com/Beckschen/TransUNet)**

These repositories provide official implementations of the baseline models used for comparison.

## Open Source Contributions
Some files in this repository have been adapted from the following open-source implementation:

```
https://github.com/bubbliiiing/unet-tf2
```

For further details, please refer to our paper. 

