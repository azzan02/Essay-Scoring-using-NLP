# Essay Score Prediction Project
This repository contains two distinct approaches to the problem of automated essay scoring (AES). Both approaches aim to predict essay scores based on text features and model learning, showcasing different methodologies to achieve the same goal.
## Overview
Automated Essay Scoring (AES) is an application of machine learning to evaluate and score essays based on various features. This repository provides:
  - **Feature-Based Model:** Uses a 32-feature approach.
  - **LSTM-Based Model:** Implements a deep learning method leveraging a Long Short-Term Memory (LSTM) neural network.
## Features
1. **Feature-Based Approach**
  - **File:** 32_features_file.ipynb
  - Description: This approach uses manually engineered features extracted from essays (such as word count, average sentence length, etc.) to predict scores using        traditional machine learning algorithms.

2. **LSTM-Based Approach**
  - **File:** lstm_file.ipynb
  - Description: This approach utilizes a deep learning model based on LSTM, which processes the raw essay text to learn contextual representations and predict the       score.

## Project Structure
```
.\
├── 32_features_file.ipynb    # Notebook for the feature-based model\
├── lstm_file.ipynb           # Notebook for the LSTM-based model\
└── README.md                 # Project documentation\
```

## Requirements
To run the notebooks, install the following dependencies:
  - Python 3.8+
  - Jupyter Notebook
  - NumPy
  - Pandas
  - Sci-Kit learn
  - TensorFlow or PyTorch (for the LSTM model)
  - NLTK or SpaCy (for text preprocessing)
Install all dependencies using pip:
```
pip install -r requirements.txt
```
## Usage
1. Clone this repository:
   ```
   git clone https://github.com/azzan02/Essay-scoring-using-NLP.git
   cd essay-score-prediction
   ```
2. Open the respective notebook **(32_features_file.ipynb or lstm_file.ipynb)** in Jupyter Notebook or Jupyter Lab.
3. Follow the instructions within the notebook to preprocess the data, train the model, and evaluate performance.

## Results
  - **Feature-Based Model:** Provides interpretability by analyzing specific features contributing to the score prediction.
  - **LSTM-Based Model:** Leverages raw text data for improved performance but requires more computational resources.

## Future Enhancements
  - Integrate additional features like semantic analysis or topic modeling.
  - Fine-tune the LSTM model using pre-trained embeddings (e.g., GloVe, BERT).
  - Add an ensemble approach combining the strengths of both models.

# License
This project is licensed under the MIT License.


