# Next Word Prediction with LSTM

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.9+" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow 2.x" />
  <img src="https://img.shields.io/badge/NLP-Next%20Word%20Prediction-5B5EA6?style=for-the-badge" alt="NLP" />
  <img src="https://img.shields.io/badge/Deep%20Learning-LSTM-8A2BE2?style=for-the-badge" alt="LSTM" />
</p>

A deep learning project for predicting the next word in a sequence using a Long Short-Term Memory (LSTM) neural network. This repository demonstrates how a model can learn patterns from text, convert words into sequences, and predict the most likely continuation for a given phrase.

This project is a beginner-friendly but practical example of natural language processing (NLP) and sequence modeling using TensorFlow and Keras.

---

## Table of Contents

- [Overview](#overview)
- [Project Features](#project-features)
- [Repository Structure](#repository-structure)
- [Technologies Used](#technologies-used)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Example Workflow](#example-workflow)
- [Model Training Concept](#model-training-concept)
- [Potential Improvements](#potential-improvements)
- [Future Scope](#future-scope)
- [License](#license)
- [Conclusion](#conclusion)

---

## Overview

Next-word prediction is a foundational task in natural language processing and text generation. The goal is to take a partial sentence or sequence of words and predict what word is likely to appear next.

For example:

- Input: `the quick brown`
- Model prediction: `fox`

This project learns from a large text corpus and builds an LSTM model that can understand sequential dependencies between words.

The workflow includes:

1. Loading text data
2. Cleaning the corpus
3. Lowercasing and filtering
4. Lemmatization and tokenization
5. Creating word sequences
6. Training an LSTM model
7. Predicting the next word

---

## Project Features

- Text-based next-word prediction
- LSTM-powered sequence modeling
- Tokenization and vocabulary creation
- Training from raw text corpus
- Sequence-to-sequence learning style for word prediction
- Notebook-based implementation for easy experimentation
- Beginner-friendly structure for NLP learning

---

## Repository Structure

```text
Next_word_pre_LSTM/
├── 1661-0.txt
├── Next_word_predicted.ipynb
├── README.md
└── .gitignore    (if added later)
```

### Files

- `1661-0.txt` — text corpus used for training the model
- `Next_word_predicted.ipynb` — notebook containing the complete implementation
- `README.md` — project documentation and usage guide

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- NLTK
- Jupyter Notebook
- Regular expressions for text cleaning

---

## How It Works

The project follows a standard natural language processing pipeline:

### 1. Data Loading
The notebook reads a text source file from the repository and stores it as a string.

### 2. Text Cleaning
The raw text is normalized to remove unwanted characters and convert everything to lowercase. This helps keep the vocabulary consistent.

### 3. Lemmatization
Words are reduced to their base forms using NLTK's `WordNetLemmatizer`, improving consistency across similar words.

### 4. Sentence and Word Tokenization
The cleaned text is split into sentences and then tokenized into words.

### 5. Sequence Creation
The model creates sequences of words such as:

```python
["the", "quick", "brown"]
```

and uses the following word as the target label.

### 6. Model Building
An LSTM network is created using Keras. This architecture is well-suited to sequential dependencies in text.

### 7. Training
The model is trained on generated word sequences, learning patterns and context from the corpus.

### 8. Prediction
After training, the model can take a phrase as input and predict the next word most likely to follow.

---

## Installation

### Prerequisites

Make sure you have the following installed:

- Python 3.9+
- Jupyter Notebook or JupyterLab
- TensorFlow
- NumPy
- pandas
- NLTK

### Install dependencies

```bash
pip install tensorflow numpy pandas nltk jupyter
```

### Download NLTK data

The notebook may use NLTK resources such as `wordnet` and `punkt_tab`. These are usually downloaded automatically during execution.

---

## Running the Project

Open the notebook in Jupyter:

```bash
jupyter notebook "Next_word_predicted.ipynb"
```

Then run all cells in order.

The notebook will:

- load the training text
- preprocess it
- build the training sequences
- create the LSTM model
- train the model
- test word predictions

---

## Example Workflow

The notebook uses code similar to:

```python
import re
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
```

### Typical processing flow

```python
text = text.lower()
text = ''.join(re.findall(r'[a-z\s]', text)).strip()

lemmatizer = WordNetLemmatizer()
lem_text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

# sentence tokenization
sentences = nltk.tokenize.sent_tokenize(lem_text)

# word tokenization
 tokenizer = Tokenizer()
 tokenizer.fit_on_texts(sentences)
 sequences = tokenizer.texts_to_sequences(sentences)
```

This forms the basis for training the neural network to predict the next word.

---

## Model Training Concept

LSTM networks are ideal for sequential data because they can remember previous information in the sequence and use it to make better predictions.

In this project:

- each word is mapped to an integer index
- sequences of words are formed as input patterns
- the model learns the relationship between previous context and the next word
- the output layer predicts the most likely next token

This is the same principle used in autocomplete engines, language models, and text generation systems.

---

## Why This Project Is Useful

This project helps illustrate key concepts in machine learning and NLP:

- representation of text as numbers
- handling sequential data
- sequence learning with recurrent neural networks
- using context to predict future words
- practical deep learning with TensorFlow

It is especially useful for students and beginners learning how language models work in practice.

---

## Potential Improvements

This project can be improved in several ways:

- Use a larger and more diverse dataset
- Add better preprocessing strategies
- Increase sequence length for richer context
- Save the trained model for reuse
- Add a prediction function for custom inputs
- Use GRU or Transformer models for comparison
- Improve vocabulary handling and rare-word filtering
- Build a web interface or API for live predictions

---

## Future Scope

This concept can be expanded into a more advanced NLP application such as:

- smart keyboard suggestions
- autocomplete engines
- sentence completion tools
- chatbot response generation
- AI-powered writing assistants
- text-generation web apps

---

## License

This repository currently does not show a dedicated license file. If this project is intended for public sharing or open-source distribution, it is recommended to add a license such as:

- MIT License
- Apache 2.0
- BSD License

Also, the dataset file `1661-0.txt` may have its own licensing restrictions depending on its original source. Please check the source dataset terms before redistribution or commercial use.

---

## Conclusion

This project is a practical and educational example of how deep learning can be used for next-word prediction. By combining text processing and LSTM-based sequence modeling, it demonstrates the core idea behind modern language models.

The repository is valuable for understanding:

- NLP preprocessing
- word embeddings and tokenization
- LSTM sequence modeling
- building prediction systems from text data

It serves as a strong foundation for learners who want to move from basic machine learning concepts into real-world natural language processing systems.

---

## Acknowledgement

This project is inspired by the fundamental principles of deep learning and language modeling. It is a great starting point for those interested in sequence learning, predictive text, and neural NLP systems.

If you want, this README can also be customized into a more polished academic, portfolio, or GitHub showcase version.
