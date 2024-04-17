# DLMI_Breast_Cancer_Segmentation
This repository is the 2024 homework for the NTU Deep Learning Medical Image course.
It aims to experiment with different state-of-the-art (SOTA) models, exploring their strengths and weaknesses.  Moreover, it compares their performances of dice score, and visualize the segmentation results for a comprehensive evaluation.
## Dataset (Breast Ultrasound Images Dataset)
Detecting breast cancer early is crucial for saving lives. Combining ultrasound imaging with deep learning significantly enhances the accuracy of identifying breast cancer stages. 
This dataset comprises 780 images, each with a resolution of 500x500 pixels in PNG format. The images are classified into three categories and each image has corresponding mask: 

- normal
- benign
- malignant
## Methodology
In this homework, I chose 9 different types of popular models: Unet, Unet++, MAnet, Linknet, FPN, PSPNet, PAN. DeepLabV3, and DeepLabV3+.
1. Unet
2. UNet++
3. MAnet
4. Linknet
5. FPN
6. PSPNet
7. PAN
8. DeepLabV3
9. DeepLabV3+
## Reference

[Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset/data)

[Model](https://github.com/qubvel/segmentation_models.pytorch)
