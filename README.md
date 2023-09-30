Sentiment Analysis of Tweets

Overview
This Python script is part of a project aimed at performing sentiment analysis on tweets to classify them as either "disaster-related" or "non-disaster." The project uses various NLP techniques, machine learning algorithms, and deep learning models to achieve this classification.

Dependencies
TensorFlow
NLTK (Natural Language Toolkit)
Pandas
NumPy
Scikit-Learn
Seaborn
Matplotlib
Keras
Transformers (Hugging Face)

Data Sources
The script loads training and test data from CSV files, typically named "train.csv" and "test.csv." 
The training data includes tweet text and corresponding target labels (0 for non-disaster, 1 for disaster-related tweets). 
The test data contains tweet text for predictions.

Data Preprocessing
The script performs extensive data preprocessing, including:
Converting text to lowercase
Removing retweets
Dropping URLs
Lemmatizing words
Removing stopwords
Vectorizing text using TF-IDF and Count Vectorization
Tokenizing and padding sequences for deep learning models

Machine Learning Models
The script includes the following machine learning models:
- Support Vector Classifier (SVC)
- Logistic Regression
- Deep Learning Model

The script uses a deep learning model with the following layers:

- Embedding layer
- Batch normalization
- Global max-pooling
- Dropout layer
- Dense layer with sigmoid activation

BERT Model
A pre-trained BERT (Bidirectional Encoder Representations from Transformers) model is also employed for sentiment analysis. 
The BERT model is fine-tuned on the tweet data for classification.

Outputs
The script generates prediction outputs in CSV format for both the machine learning and BERT models, which can be submitted 
for evaluation in a Kaggle competition or any other sentiment analysis task.

