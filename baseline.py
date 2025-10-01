###an equivalent to the train_test.py file but for baseline model a tf-idf model
###made with github copilot with minor edits using the above line as prompt. Secondary file used only as a comparison to the real model

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from tqdm import tqdm
import joblib
import os
from data import CodeRecordDataset
from config import ExperimentConfig


def calculate_metrics(preds, true_labels):
    preds = (preds > 0.5).astype(np.float32)
    true_labels = true_labels.astype(np.float32)
    tp = (preds * true_labels).sum(axis=0)
    prec = tp / (preds.sum(axis=0) + 1e-8)
    rec = tp / (true_labels.sum(axis=0) + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    accuracy = (preds == true_labels).mean(axis=0)
    return f1, prec, rec, accuracy


def train(config, train_dataset, val_dataset):
    # Extract text and labels from datasets
    train_texts = [item['description']['raw_text'] for item in train_dataset]
    train_labels = np.array([item['labels'].numpy() for item in train_dataset])
    
    val_texts = [item['description']['raw_text'] for item in val_dataset]
    val_labels = np.array([item['labels'].numpy() for item in val_dataset])

    # Create and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=10000)
    train_features = vectorizer.fit_transform(train_texts)
    val_features = vectorizer.transform(val_texts)

    # Train logistic regression model
    base_lr = LogisticRegression(max_iter=1000)
    model = MultiOutputClassifier(base_lr)
    model.fit(train_features, train_labels)

    # Evaluate on validation set
    val_preds = model.predict_proba(val_features)
    val_preds = np.array([pred[:, 1] for pred in val_preds]).T
    f1, prec, rec, accuracy = calculate_metrics(val_preds, val_labels)

    print('Validation results:')
    for i, label in enumerate(config.data_labels):
        print(f'Tag {label}: F1 {f1[i]:.4f} (precision {prec[i]:.4f}, recall {rec[i]:.4f}) accuracy: {accuracy[i]:.4f}')

    # Save the model
    if not os.path.exists('runs/tfidf_lr'):
        os.makedirs('runs/tfidf_lr')
    joblib.dump(vectorizer, 'runs/tfidf_lr/vectorizer.joblib')
    joblib.dump(model, 'runs/tfidf_lr/model.joblib')

    return f1, prec, rec, accuracy


@torch.no_grad()
def test(config, test_dataset):
    # Load saved model
    vectorizer = joblib.load('runs/tfidf_lr/vectorizer.joblib')
    model = joblib.load('runs/tfidf_lr/model.joblib')

    # Extract text and labels from test dataset
    test_texts = [item['description']['raw_text'] for item in test_dataset]
    test_labels = np.array([item['labels'].numpy() for item in test_dataset])

    # Transform text and predict
    test_features = vectorizer.transform(test_texts)
    test_preds = model.predict_proba(test_features)
    test_preds = np.array([pred[:, 1] for pred in test_preds]).T

    # Calculate metrics
    f1, prec, rec, accuracy = calculate_metrics(test_preds, test_labels)
    weights = test_labels.sum(axis=0) / test_labels.sum()

    print(f'Overall accuracy: {(weights*accuracy).sum():.4f}, overall F1: {(f1*weights).sum():.4f}')
    print('Per-tag results:')
    for i, label in enumerate(config.data_labels):
        print(f'Tag {label}: test F1 {f1[i]:.4f} (precision {prec[i]:.4f}, recall {rec[i]:.4f}) accuracy: {accuracy[i]:.4f}')

    return f1, prec, rec, accuracy, weights


def main():
    config = ExperimentConfig()
    full_dataset = CodeRecordDataset(config)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [0.7, 0.15, 0.15])

    print("Training baseline model...")
    train(config, train_dataset, val_dataset)

    print("\nTesting baseline model...")
    test(config, test_dataset)


if __name__ == "__main__":
    main()
