This repository is just to show and teach some Sentiment Analysis methods, from basic NLP to transformers.

# Sentiment Analysis E-Commerce Product Reviews NLP

This is a classic NLP problem dealing with data from an e-commerce store focusing on women's clothing. Each record in the dataset is a customer review which consists of the review title, text description and a recommendation 0 or 1) for a product amongst other features

Main Objective: Leverage the review text attributes and other features if needed to predict the recommendation (classification)

![](https://media-cldnry.s-nbcnews.com/image/upload/newscms/2020_11/1548335/clothes-corona-today-main-200313.jpg)

In this case you will want to segment the image, i.e., each pixel of the image is given a label. 

Thus, the task of image segmentation is to train a neural network to output a pixel-wise mask of the image. This helps in understanding the image at a much lower level, i.e., the pixel level. 

Image segmentation has many applications in medical imaging, self-driving cars and satellite imaging to name a few.

Techniques applied:
- Data Augmentation
- U-Net model
  - Encoder: MobileNetV2 model (Transfer learning)
  - Decoder: [Pix2pix](https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py)
---

### Task:

In this competition, you’re challenged to develop an algorithm that automatically removes the photo studio background. This will allow Carvana to superimpose cars on a variety of backgrounds.

---
### Methodology

  - **Daa Augmentation**
    - The following code performs a simple augmentation of flipping an image. 
    - In addition,  image is normalized to [0,1].

<p align="center">
<image src="Notebooks/images/examples0.png" width=600px/>
</p>

<p align="center">
<image src="Notebooks/images/examples01.png" width=600px/>
</p>


   - **U-Net Model**
    - Encoder: MobileNetV2 model (Transfer learning)
    - Decoder: [Pix2pix](https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py)

The model being used here is a modified U-Net. A U-Net consists of an encoder (downsampler) and decoder (upsampler). In-order to learn robust features, and reduce the number of trainable parameters, a pretrained model can be used as the encoder. Thus, the encoder for this task will be a pretrained MobileNetV2 model, whose intermediate outputs will be used, and the decoder will be the upsample block already implemented in TensorFlow Examples in the [Pix2pix tutorial](https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py). 

The reason to output three channels is because there are three possible labels for each pixel. Think of this as multi-classification where each pixel is being classified into three classes.

<p align="center">
<image src="Notebooks/images/model.png" width=800px/>
</p>

  - **Segmentation before Training**
 
 <p align="center">
<image src="Notebooks/images/segmentation_before_training.png" width=800px/>
</p>

  - **Segmentation - Training for 3 Epochs**
    - We are going to use Dice Coefficient (F1 Score) to validate our model. [Model metrics](https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2)

**Just with 3 epochs: val_loss: 0.0155 - val_dice_coef: 0.9845 - val_binary_accuracy: 0.9941**

<p align="center">
<image src="Notebooks/images/loss_value_epochs.png" width=500px/>
</p>

### Examples of results after training

<p align="center">
<image src="Notebooks/images/predictions0.png" width=800px/>
</p>

<p align="center">
<image src="Notebooks/images/predictions1.png" width=800px/>
</p>

<p align="center">
<image src="Notebooks/images/predictions2.png" width=800px/>
</p>

<p align="center">
<image src="Notebooks/images/predictions3.png" width=800px/>
</p>

<p align="center">
<image src="Notebooks/images/predictions4.png" width=800px/>
</p>
