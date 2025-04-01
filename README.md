# gland-msconnet
Gland-MSConNet: A semantic segmentation Model for Cotton Leaf Pigment Gland Detection and Phenotypic Quantification
![Figure_2](https://github.com/user-attachments/assets/3ffdbf2d-8929-4c83-8cbf-1a7746d952fa)

Getting Started

1.Installation:
Before running the code, ensure you have the following dependencies installed:
pip install -r requirements.txt

2.Data Preparation:
The dataset is available here:https://pan.baidu.com/s/1-a8_EDrq0sIySpnlFbvAlg?pwd=luck (extraction code: luck).

3.Data Format:
Input: RGB image (e.g., 256×256 resolution).
Output: Binary segmentation mask (grayscale).

4.Training:
To train a model, run train.py.

5.Evaluation:
To validate the model, run get_miou.py. 

6.Comparison Models:
For benchmarking, we compare Gland-MSConNet with the following models:

  unet：https://github.com/bubbliiiing/unet-pytorch
  
  deeplabv3+：https://github.com/bubbliiiing/deeplabv3-plus-pytorch
  
  pspnet：https://github.com/bubbliiiing/pspnet-pytorch
  
  transunet：https://github.com/Beckschen/TransUNet
  
These repositories provide the official implementations of the baseline models used for comparison.

7.Open Source:
A few files in this repository are modified from the following open-source implementations:
https://github.com/bubbliiiing/unet-tf2

For more details, please refer to our paper. 
