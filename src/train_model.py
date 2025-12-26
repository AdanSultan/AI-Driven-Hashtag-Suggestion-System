"""
AI-Driven Hashtag Suggestion System - Model Training Script

This script trains a machine learning model to predict hashtags for tweets.
It includes cross-validation, confusion matrix visualization, and comprehensive metrics.
"""

import pandas as pd
import re
import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "ai_hashtag_suggestion_dataset_10k.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.txt")
MODEL_PATH = os.path.join(RESULTS_DIR, "hashtag_model.pkl")
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")


def clean_text(text):
    """
    Clean and preprocess tweet text.
    
    Args:
        text: Raw tweet text
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)     # Remove mentions
    text = re.sub(r"[^a-z\s]", "", text) # Remove special characters
    text = re.sub(r"\s+", " ", text)     # Remove extra whitespace
    return text.strip()


def load_data(filepath):
    """Load and validate the dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    required_columns = ["tweet_text", "hashtag_label"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove any rows with missing values
    initial_count = len(df)
    df = df.dropna(subset=required_columns)
    if len(df) < initial_count:
        print(f"⚠️  Removed {initial_count - len(df)} rows with missing values")
    
    return df


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """Generate and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - Hashtag Classification', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def train_model():
    """Main training function."""
    print("\n" + "=" * 60)
    print("AI-DRIVEN HASHTAG SUGGESTION SYSTEM - TRAINING")
    print("=" * 60)
    
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load data
    print("\nLoading dataset...")
    df = load_data(DATA_PATH)
    print(f"   Total samples: {len(df)}")
    print(f"   Categories: {df['hashtag_label'].nunique()}")
    
    # Clean text
    print("\n🧹 Cleaning text data...")
    df["tweet_text"] = df["tweet_text"].apply(clean_text)
    
    # Remove empty texts after cleaning
    df = df[df["tweet_text"].str.len() > 0]
    
    X = df["tweet_text"]
    y = df["hashtag_label"]
    
    # Get unique labels sorted
    labels = sorted(y.unique())
    
    # Train-test split
    print("\n Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Model pipeline
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=2000, random_state=42))
    ])
    
    # Cross-validation
    print("\n Performing 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"   CV Scores: {cv_scores.round(4)}")
    print(f"   CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train final model
    print("\n  Training final model...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    
    # Generate confusion matrix
    print("\n Generating confusion matrix...")
    plot_confusion_matrix(y_test, y_pred, labels, CONFUSION_MATRIX_PATH)
    
    # Save metrics
    print("\n Saving metrics...")
    with open(METRICS_PATH, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("AI-DRIVEN HASHTAG SUGGESTION SYSTEM\n")
        f.write("MODEL EVALUATION METRICS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("DATASET INFO:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"Categories: {len(labels)}\n\n")
        
        f.write("CROSS-VALIDATION (5-Fold):\n")
        f.write("-" * 40 + "\n")
        f.write(f"Scores: {cv_scores.round(4)}\n")
        f.write(f"Mean Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n\n")
        
        f.write("TEST SET METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        
        f.write("WEIGHTED AVERAGES (accounts for class frequency):\n")
        f.write(f"  Precision: {precision_weighted:.4f}\n")
        f.write(f"  Recall:    {recall_weighted:.4f}\n")
        f.write(f"  F1-Score:  {f1_weighted:.4f}\n\n")
        
        f.write("MACRO AVERAGES (treats all classes equally):\n")
        f.write(f"  Precision: {precision_macro:.4f}\n")
        f.write(f"  Recall:    {recall_macro:.4f}\n")
        f.write(f"  F1-Score:  {f1_macro:.4f}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("PER-CLASS PERFORMANCE:\n")
        f.write("=" * 60 + "\n")
        f.write(classification_report(y_test, y_pred))
    
    print(f"   Metrics saved to: {METRICS_PATH}")
    
    # Save model
    print("\n Saving model...")
    joblib.dump(model, MODEL_PATH)
    print(f"   Model saved to: {MODEL_PATH}")
    
    # Print summary
    print("\n" + "=" * 60)
    print(" TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n RESULTS SUMMARY:")
    print(f"   Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"   Test Set Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\n   Weighted -> P: {precision_weighted:.4f} | R: {recall_weighted:.4f} | F1: {f1_weighted:.4f}")
    print(f"   Macro    -> P: {precision_macro:.4f} | R: {recall_macro:.4f} | F1: {f1_macro:.4f}")
    print("\n" + "=" * 60)
    
    return model


if __name__ == "__main__":
    try:
        train_model()
    except FileNotFoundError as e:
        print(f"\n ERROR: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n UNEXPECTED ERROR: {e}")
        sys.exit(1)
