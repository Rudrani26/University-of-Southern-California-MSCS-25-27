import pandas as pd
import numpy as np
import nltk
import re
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 1. Dataset Preparation
df = pd.read_csv('data.tsv', sep='\t', on_bad_lines='skip', low_memory=False)

# Converting 'star_rating' to numeric and any invalid or non-numeric values will be turned into NaN
df['star_rating'] = pd.to_numeric(df['star_rating'], errors='coerce')

# Dropping rows where 'star_rating' or 'review_body' is missing/invalid
df.dropna(subset=['star_rating', 'review_body'], inplace=True)

# Converting star ratings to integers (e.g., 1.0 -> 1)
# This makes it easier to compare rating values
df['star_rating'] = df['star_rating'].astype(int)

# BEFORE dropping the neutral reviews
# Positive reviews (ratings greater than 3)
positive_count = len(df[df['star_rating'] > 3])
# Negative reviews (ratings 2 or lower)
negative_count = len(df[df['star_rating'] <= 2])
# Neutral reviews (rating equal to 3)
neutral_count = len(df[df['star_rating'] == 3])

print(f"Positive reviews: {positive_count}")
print(f"Negative reviews: {negative_count}")
print(f"Neutral reviews: {neutral_count}")

# Drop all reviews with a star rating of 3
# Neutral reviews are excluded to simplify binary classification
df = df[df['star_rating'] != 3]

# Creating Binary Labels
# 1 = Positive (>3), 0 = Negative (<=2)
# Since we already dropped 3s, everything remaining is either >3 or <=2
df['label'] = df['star_rating'].apply(lambda x: 1 if x > 3 else 0)

# Downsampling
# Randomly selecting 100,000 positive and 100,000 negative reviews
pos_reviews = df[df['label'] == 1].sample(n=100000, random_state=42)
neg_reviews = df[df['label'] == 0].sample(n=100000, random_state=42)

dataset = pd.concat([pos_reviews, neg_reviews]).sample(frac=1, random_state=42).reset_index(drop=True)

# Average Length (Before Cleaning)
dataset['len_pre_clean'] = dataset['review_body'].apply(len)
print(f"Average length before cleaning: {dataset['len_pre_clean'].mean():.4f}")

# Contraction Dictionary
contraction_dict = {
    "won't": "will not",
    "can't": "cannot",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
    "i'm": "i am",
    "aren't": "are not",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "i would",
    "i'll": "i will",
    "i've": "i have",
    "isn't": "is not",
    "it's": "it is",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who're": "who are",
    "who's": "who is",
    "who've": "who have",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have"
}

def expand_contractions(text):
    # Using regex to ensure we match whole words or suffixes correctly
    # specific patterns like "won't" need to be checked before "n't"
    for contraction, expansion in contraction_dict.items():
        text = re.sub(r"\b" + re.escape(contraction) + r"\b", expansion, text)
    return text

def clean_text(text):
    # converting to lowercase
    text = str(text).lower()

    # Removing HTML
    if "<" in text:
        text = BeautifulSoup(text, "html.parser").get_text()

    # Removing URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Expand Contractions
    text = expand_contractions(text)

    # Removing non-alphabetical characters
    # Keeping spaces so words don't merge
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Removing extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

dataset['cleaned_reviews'] = dataset['review_body'].apply(clean_text)

# Average Length (After Cleaning)
dataset['len_post_clean'] = dataset['cleaned_reviews'].apply(len)
print(f"Average length after cleaning: {dataset['len_post_clean'].mean():.4f}")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    words = text.split()
    # Remove stopwords and lemmatize
    filtered_words = [
        lemmatizer.lemmatize(word.lower())
        for word in words
        if word.lower() not in stop_words
    ]
    return " ".join(filtered_words)

dataset['preprocessed_reviews'] = dataset['cleaned_reviews'].apply(preprocess_text)

# Length BEFORE preprocessing (characters)
dataset['len_before'] = dataset['cleaned_reviews'].apply(len)

# Length AFTER preprocessing (characters)
dataset['len_after'] = dataset['preprocessed_reviews'].apply(len)

print(f"Average length before preprocessing: {dataset['len_before'].mean():.4f}")
print(f"Average length after preprocessing:  {dataset['len_after'].mean():.4f}")

# 4. Feature Extraction
vectorizer = TfidfVectorizer(ngram_range=(2, 2))

# Fit and transform
X = vectorizer.fit_transform(dataset['preprocessed_reviews'])
y = dataset['label']

# 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5-8. Model Training and Evaluation
def train_and_eval(model, model_name):
    # Train
    model.fit(X_train, y_train)

    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    def print_metrics(split_name, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f"{model_name} {split_name} Accuracy: {acc:.4f}")
        print(f"{model_name} {split_name} Precision: {prec:.4f}")
        print(f"{model_name} {split_name} Recall: {rec:.4f}")
        print(f"{model_name} {split_name} F1-score: {f1:.4f}")

    print_metrics("Training", y_train, y_train_pred)
    print_metrics("Testing", y_test, y_test_pred)

perceptron = Perceptron(random_state=42)
train_and_eval(perceptron, "Perceptron")
svm = LinearSVC(random_state=42)
train_and_eval(svm, "SVM")
logreg = LogisticRegression(random_state=42, max_iter=1000)
train_and_eval(logreg, "Logistic Regression")
mnb = MultinomialNB()
train_and_eval(mnb, "Naive Bayes")