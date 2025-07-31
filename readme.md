# Deep Learning Course Workshops

\<div align="center"\>
\<img src="[https://img.shields.io/badge/Python-3776AB?style=for-the-badge\&logo=python\&logoColor=white](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)" alt="Python"\>
\<img src="[https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge\&logo=pytorch\&logoColor=white](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)" alt="PyTorch"\>
\<img src="[https://img.shields.io/badge/NLP-4CAF50?style=for-the-badge\&logoColor=white](https://img.shields.io/badge/NLP-4CAF50?style=for-the-badge&logoColor=white)" alt="NLP"\>
\<img src="[https://img.shields.io/badge/Deep%20Learning-FF6F61?style=for-the-badge\&logoColor=white](https://img.shields.io/badge/Deep%20Learning-FF6F61?style=for-the-badge&logoColor=white)" alt="Deep Learning"\>
\</div\>

This repository contains a series of hands-on workshops designed for a deep learning course. Each workshop focuses on a specific topic, providing practical experience with fundamental concepts and popular frameworks in the field.

## 📚 Table of Contents

1.  [Workshop 1: NumPy Fundamentals](https://www.google.com/search?q=%23workshop-1-numpy-fundamentals)
2.  [Workshop 2: Introduction to Pandas](https://www.google.com/search?q=%23workshop-2-introduction-to-pandas)
3.  [Workshop 3: Introduction to PyTorch](https://www.google.com/search?q=%23workshop-3-introduction-to-pytorch)
4.  [Workshop 4: Convolutional Neural Networks (CNNs)](https://www.google.com/search?q=%23workshop-4-convolutional-neural-networks-cnns)
5.  [Workshop 5: NLP and Recurrent Neural Networks (RNNs)](https://www.google.com/search?q=%23workshop-5-nlp-and-recurrent-neural-networks-rnns)
6.  [Workshop 6: Transformers from Scratch](https://www.google.com/search?q=%23workshop-6-transformers-from-scratch)
7.  [Workshop 7: Language Models and Parameter-Efficient Fine-Tuning](https://www.google.com/search?q=%23workshop-7-language-models-and-parameter-efficient-fine-tuning)
8.  [Workshop 8: Generative Models and LoRA Fine-Tuning](https://www.google.com/search?q=%23workshop-8-generative-models-and-lora-fine-tuning)

-----

## Workshop 1: NumPy Fundamentals ([WS\_1.ipynb](https://www.google.com/search?q=./WS_1.ipynb))

This workshop covers the basics of NumPy, an essential library for numerical computing in Python and a foundational block for deep learning.

### 🔍 Key Topics:

  - Calculating mean, standard deviation, max, min, and median of a matrix
  - Performing zero-mean normalization
  - Traversing matrices and calculating correlation coefficients between vectors
  - Counting occurrences of a number in a NumPy array

-----

## Workshop 2: Introduction to Pandas ([WS\_2.ipynb](https://www.google.com/search?q=./WS_2.ipynb))

This workshop introduces pandas, a powerful library for data manipulation and analysis.

### 🔍 Key Topics:

  - Introduction to Series and DataFrame objects
  - Data indexing and selection
  - Accessing data using `loc` and `iloc`
  - Understanding Ufuncs and index preservation

-----

## Workshop 3: Introduction to PyTorch ([workshop\_3.ipynb](https://www.google.com/search?q=./workshop_3.ipynb))

This workshop covers the fundamental concepts of the PyTorch deep learning framework.

### 🔍 Key Topics:

  - Creating and manipulating tensors
  - Basic tensor operations (addition, subtraction, division)
  - Leveraging the GPU for tensor computations
  - Understanding computation graphs and backpropagation

-----

## Workshop 4: Convolutional Neural Networks (CNNs) ([workshop\_4.ipynb](https://www.google.com/search?q=./workshop_4.ipynb))

This workshop dives into the basics of Convolutional Neural Networks (CNNs), widely used for image recognition and processing tasks.

### 🔍 Key Topics:

  - Convolutional operations, pooling, and activation functions
  - Fully connected layers and Softmax for classification
  - Implementing a simple CNN using the CIFAR-10 dataset
  - Exploring different optimizers

-----

## Workshop 5: NLP and Recurrent Neural Networks (RNNs)

This section is divided into two parts, focusing on Natural Language Processing (NLP) and Recurrent Neural Networks (RNNs).

### Part 1: Toxic Comment Classification ([WS5\_T1.ipynb](https://www.google.com/search?q=./WS5_T1.ipynb))

  - Learn how to use an RNN to classify comments as toxic or non-toxic

### Part 2: Time Series Forecasting ([WS5\_T2\_MyModel.ipynb](https://www.google.com/search?q=./WS5_T2_MyModel.ipynb), [WS5\_T2\_QuestionModel.ipynb](https://www.google.com/search?q=./WS5_T2_QuestionModel.ipynb))

  - Use a Long Short-Term Memory (LSTM) model to predict Bitcoin prices
  - Analyze how the look-back period affects model performance

-----

## Workshop 6: Transformers from Scratch ([WS\_6.ipynb](https://www.google.com/search?q=./WS_6.ipynb))

This workshop guides you through implementing a Transformer model from the ground up, a model architecture that has revolutionized NLP.

### 🔍 Key Topics:

  - Building a basic Transformer model using PyTorch
  - Analyzing how the temperature parameter affects text generation

-----

## Workshop 7: Language Models and Parameter-Efficient Fine-Tuning ([WS\_7.ipynb](https://www.google.com/search?q=./WS_7.ipynb))

This workshop explores advanced topics in language modeling and efficient model adaptation techniques.

### 🔍 Key Topics:

  - The three main categories of language models:
      - Encoder-only
      - Encoder-decoder
      - Decoder-only
  - Adapting models to new tasks using:
      - Zero-shot learning
      - Few-shot learning
  - Practical example of zero-shot learning with the T5 model

-----

## Workshop 8: Generative Models and LoRA Fine-Tuning ([WS\_8/WorkShop\_8.ipynb](https://www.google.com/search?q=./WS_8/WorkShop_8.ipynb))

This final workshop explores generative models, including GANs, VAEs, and Diffusion models. The main task involves fine-tuning a pre-trained **Stable Diffusion** model on a medical dataset using **Low-Rank Adaptation (LoRA)**. The task notebooks, with models trained for 1 and 5 epochs, can be found in the `WS_8/Task` directory.

### 🔍 Key Topics:

  - **Conditional GANs (CGANs):** Building a CGAN from scratch to generate MNIST digits based on class labels.
  - **Variational Autoencoders (VAEs):** Implementing Encoder, Decoder, and the reparameterization trick to build a VAE and visualize its latent space.
  - **Stable Diffusion:** An introduction to the components and workflow of modern text-to-image diffusion models.
  - **LoRA Fine-Tuning Task:** A self-supervised task to fine-tune Stable Diffusion on chest X-ray images and clinical notes by implementing custom `LoRALinear` layers to adapt the model efficiently.

-----

## 📜 License

This project is part of academic coursework. Please refer to individual workshop notebooks for specific licensing information.

\<div align="center"\>
\<strong\>Happy Learning\! 🚀\</strong\>
\</div\>
