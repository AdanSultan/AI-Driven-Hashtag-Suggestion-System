"""
AI-Driven Hashtag Suggestion System - Prediction Script

This script loads the trained model and provides hashtag suggestions for tweets.
"""

import joblib
import re
import os
import sys

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "results", "hashtag_model.pkl")

# Global model variable
_model = None


def load_model():
    """Load the trained model with error handling."""
    global _model
    
    if _model is not None:
        return _model
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Please run 'python src/train_model.py' first to train the model."
        )
    
    try:
        _model = joblib.load(MODEL_PATH)
        return _model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


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


def suggest_hashtag(tweet, return_probabilities=False):
    """
    Suggest a hashtag for the given tweet.
    
    Args:
        tweet: The tweet text to analyze
        return_probabilities: If True, also return confidence scores
        
    Returns:
        Predicted hashtag (str) or tuple of (hashtag, probabilities) if return_probabilities=True
    """
    if not tweet or not tweet.strip():
        raise ValueError("Tweet text cannot be empty")
    
    model = load_model()
    cleaned_tweet = clean_text(tweet)
    
    if not cleaned_tweet:
        raise ValueError("Tweet contains no valid text after cleaning")
    
    prediction = model.predict([cleaned_tweet])[0]
    
    if return_probabilities:
        probabilities = model.predict_proba([cleaned_tweet])[0]
        classes = model.classes_
        prob_dict = dict(zip(classes, probabilities))
        # Sort by probability descending
        prob_dict = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
        return prediction, prob_dict
    
    return prediction


def suggest_top_hashtags(tweet, top_n=3):
    """
    Suggest top N hashtags with their confidence scores.
    
    Args:
        tweet: The tweet text to analyze
        top_n: Number of top predictions to return
        
    Returns:
        List of tuples (hashtag, probability)
    """
    _, probabilities = suggest_hashtag(tweet, return_probabilities=True)
    return list(probabilities.items())[:top_n]


def interactive_mode():
    """Run interactive prediction mode."""
    print("\n" + "=" * 50)
    print(" AI HASHTAG SUGGESTION SYSTEM")
    print("=" * 50)
    print("Enter a tweet to get hashtag suggestions.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    try:
        load_model()
        print(" Model loaded successfully!\n")
    except FileNotFoundError as e:
        print(f" {e}")
        return
    except Exception as e:
        print(f" Error loading model: {e}")
        return
    
    while True:
        try:
            tweet = input("Enter tweet: ").strip()
            
            if tweet.lower() in ['quit', 'exit', 'q']:
                print("\n Goodbye!")
                break
            
            if not tweet:
                print("  Please enter some text.\n")
                continue
            
            # Get top 3 suggestions
            top_hashtags = suggest_top_hashtags(tweet, top_n=3)
        
            print("\n Suggested Hashtags:")
            print("-" * 30)
            for i, (hashtag, prob) in enumerate(top_hashtags, 1):
                confidence = "🟢" if prob > 0.5 else "🟡" if prob > 0.2 else "🔴"
                print(f"   {i}. #{hashtag} {confidence} ({prob*100:.1f}%)")
            print()
            
        except ValueError as e:
            print(f"⚠️  {e}\n")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break


if __name__ == "__main__":
    interactive_mode()
