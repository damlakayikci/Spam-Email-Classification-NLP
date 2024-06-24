# Spam Email Classification Using NLP

## Overview

This project focuses on classifying emails as spam or not spam using Natural Language Processing (NLP) techniques. The repository includes code for training models, evaluating their performance, and a dataset for experimentation.

## Contents

- `models/`: Directory containing trained models.
- `FFNN.py`: Script for training a Feedforward Neural Network.
- `NLP - Project 1_4.pdf`: Project documentation and analysis.
- `Oppositional_thinking_analysis_dataset.json`: Dataset used for training and evaluation.
- `main.py`: Main script for running the classification.
- `utils.py`: Utility functions for data processing and model evaluation.

## Requirements

- Python 3.x
- Necessary libraries listed in `requirements.txt` 

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/damlakayikci/Spam-Email-Classification-NLP.git
   cd src

## Usage
Use one of the following scripts to run the code

1. To train and run the Naive Bayes model:

   ``` python main.py nb```
   
2. To train and run the Feedforward Neural Network (FFNN):

    ``` python main.py ffnn```
3. To print statistics and plot graphs:

    ``` python main.py stats```
4. To find the Pointwise Mutual Information (PMI) of 10 random words and print the most similar words:

    ``` python main.py pmi```


