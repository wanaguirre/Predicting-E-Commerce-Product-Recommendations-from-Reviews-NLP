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

<p align="center">
<image src="Notebooks/images/1.png" width=500px/>
</p>

  - **Experiment 1: Basic NLP Count based Features & Age, Feedback Count**
    - Use of basic text based features, that sometimes are helpful for improving text classification models. 
      - Word Count: total number of words in the documents
      - Character Count: total number of characters in the documents
      - Average Word Density: average length of the words used in the documents
      - Puncutation Count: total number of punctuation marks in the documents
      - Upper Case Count: total number of upper count words in the documents
      - Title Word Count: total number of proper case (title) words in the documents
    - Training a Logistic Regression Model

<p align="center">
<image src="Notebooks/images/2.png" width=500px/>
</p>

  - **Experiment 2: Features from Sentiment Analysis**
    - Leveraging Text Sentiment
 
 Reviews are pretty subjective, opinionated and people often express stong emotions, feelings. This makes it a classic case where the text documents here are a good candidate for extracting sentiment as a feature.

The general expectation is that highly rated and recommended products (label 1) should have a positive sentiment and products which are not recommended (label 0) should have a negative sentiment.

TextBlob is an excellent open-source library for performing NLP tasks with ease, including sentiment analysis. It also an a sentiment lexicon (in the form of an XML file) which it leverages to give both polarity and subjectivity scores.

The polarity score is a float within the range [-1.0, 1.0].
The subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective.

<p align="center">
<image src="Notebooks/images/3.png" width=500px/>
</p>

  - **Text Pre-processing and Wrangling**
    - We want to extract some specific features based on standard NLP feature engineering models like the classic Bag of Words model. For this we need to clean and pre-process our text data. We will build a simple text pre-processor here since the main intent is to look at feature engineering strategies.

      We will focus on:

      - Text Lowercasing
      - Removal of contractions
      - Removing unnecessary characters, numbers and symbols
      - Stopword removal

  - **Experiment 3: Modeling based on Bag of Words based Features - 1-grams**

This is perhaps the most simple vector space representational model for unstructured text. A vector space model is simply a mathematical model to represent unstructured text (or any other data) as numeric vectors, such that each dimension of the vector is a specific feature\attribute.

The bag of words model represents each text document as a numeric vector where each dimension is a specific word from the corpus and the value could be its frequency in the document, occurrence (denoted by 1 or 0) or even weighted values.

The model’s name is such because each document is represented literally as a ‘bag’ of its own words, disregarding word orders, sequences and grammar.

<p align="center">
<image src="Notebooks/images/4.png" width=500px/>
</p>

  - **Experiment 4: Modeling with Bag of Words based Features - 2-grams**

<p align="center">
<image src="Notebooks/images/5.png" width=500px/>
</p>

  - **Experiment 5: Adding Bag of Words based Features - 3-grams**

<p align="center">
<image src="Notebooks/images/6.png" width=500px/>
</p>

  - **Experiment 6: Adding Bag of Words based Features - 3-grams with Feature Selection**
    - We drop all words \ n-grams which occur less than 3 times in all documents.

<p align="center">
<image src="Notebooks/images/7.png" width=500px/>
</p>

  - **Experiment 7: Combining Bag of Words based Features - 3-grams with Feature Selection and the Structured Features**
    - Let's combine our sparse BOW feature matrices with our structured features from earlier. To do this concatenation, We do need to convert those structured features into sparse format.

<p align="center">
<image src="Notebooks/images/8.png" width=500px/>
</p>

  - **Experiment 8: Modeling on FastText Averaged Document Embeddings**
    - Build the FastText embedding model
    - Generate document level embeddings

Word embedding models give us an embedding for each word, how can we use it for downstream ML\DL tasks? one way is to flatten it or use sequential models. A simpler approach is to average all word embeddings for words in a document and generate a fixed-length document level embedding

<p align="center">
<image src="Notebooks/images/9.png" width=500px/>
</p>

  - **Experiment 9: Combine FastText Vectors + Structured Features and build a model**

<p align="center">
<image src="Notebooks/images/10.png" width=500px/>
</p>

  - **Experiment 10: Train Classfier with CNN + FastText Embeddings**
    - FastText Embeddings
    - Convolutional Neural Network

<p align="center">
<image src="Notebooks/images/11.png" width=500px/>
</p>

  - **Experiment 11: Train Classfier with LSTM + FastText Embeddings**

<p align="center">
<image src="Notebooks/images/12.png" width=500px/>
</p>

  - **Experiment 12: Train Classfier with NNLM Universal Embedding Model**

<p align="center">
<image src="Notebooks/images/13.png" width=500px/>
</p>

  - **Experiment 13: Train Classfier with BERT**
    - Train and Evaluate your BERT model using transformers

<p align="center">
<image src="Notebooks/images/14.png" width=500px/>
</p>
