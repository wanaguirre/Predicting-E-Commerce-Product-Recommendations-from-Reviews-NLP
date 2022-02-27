This repository is just to show and teach some Sentiment Analysis methods, from basic NLP to transformers.

# Sentiment Analysis E-Commerce Product Reviews NLP

This is a classic NLP problem dealing with data from an e-commerce store focusing on women's clothing, and based on the customer reviews we have to predict if each of them was a positive or negative recomendation.

![](https://traid.org.uk/wp-content/uploads/2019/02/1800_eac.jpg)

To solve this problem we used different techniques just to see how each of them works and which could be the better approach.

Techniques applied:
- Experiment 1: Basic NLP Count based Features & Age, Feedback Count
- Experiment 2: Features from Sentiment Analysis
- Experiment 3: Modeling based on Bag of Words based Features - 1-grams
- Experiment 4: Modeling with Bag of Words based Features - 2-grams
- Experiment 5: Adding Bag of Words based Features - 3-grams
- Experiment 6: Adding Bag of Words based Features - 3-grams with Feature Selection
- Experiment 7: Combining Bag of Words based Features - 3-grams with Feature Selection and the Structured Features
- Experiment 8: Modeling on FastText Averaged Document Embeddings
- Experiment 9: Combine FastText Vectors + Structured Features and build a model
- Experiment 10: Train Classfier with **CNN** + FastText Embeddings & Evaluate Performance on Test Data
- Experiment 11: Train Classfier with **LSTM** + FastText Embeddings & Evaluate Performance on Test Data
- Experiment 12: Train Classfier with **NNLM Universal Embedding Model**
- Experiment 13: Train Classfier with **BERT**

---

### Task:

Classify the customer's reviews.

The data is available at https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews from where you can download it.

---
### Methodology

  - **Dataset review**
    -  Merge all review text attributes (title, text description) into one attribute
    -  Subset out columns of interest
    -  Remove all records with no review text
    -  Build train and test datasets

  - **Experiment 1: Basic NLP Count based Features & Age, Feedback Count**




  - **Experiment 2: Features from Sentiment Analysis**





  - **Experiment 3: Modeling based on Bag of Words based Features - 1-grams**




  - **Experiment 4: Modeling with Bag of Words based Features - 2-grams**




  - **Experiment 5: Adding Bag of Words based Features - 3-grams**




  - **Experiment 6: Adding Bag of Words based Features - 3-grams with Feature Selection**




  - **Experiment 7: Combining Bag of Words based Features - 3-grams with Feature Selection and the Structured Features**




  - **Experiment 8: Modeling on FastText Averaged Document Embeddings**




  - **Experiment 9: Combine FastText Vectors + Structured Features and build a model**




  - **Experiment 10: Train Classfier with CNN + FastText Embeddings & Evaluate Performance on Test Data**




  - **Experiment 11: Train Classfier with LSTM + FastText Embeddings & Evaluate Performance on Test Data**




  - **Experiment 12: Train Classfier with NNLM Universal Embedding Model**




  - **Experiment 13: Train Classfier with BERT**




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
