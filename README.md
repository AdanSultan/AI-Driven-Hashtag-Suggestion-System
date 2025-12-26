#  AI-Driven Hashtag Suggestion System

An intelligent machine learning system that automatically suggests relevant hashtags for tweets/social media posts using Natural Language Processing (NLP) and classification algorithms.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Categories](#-categories)
- [How It Works](#-how-it-works)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [License](#-license)

## Overview

This project uses machine learning to analyze tweet text and predict the most appropriate hashtag category. It's trained on **20,000+ diverse samples** across **20 different categories**, making it suitable for real-world social media content classification.

## Features

-  **High Accuracy**: ~75% accuracy on real-world tweets with cross-validated results
-  **20 Categories**: Covers business, technology, sports, entertainment, and more
-  **Cross-Validation**: 5-fold stratified cross-validation for reliable metrics
-  **Confusion Matrix**: Visual representation of model performance
-  **Top-N Predictions**: Get multiple hashtag suggestions with confidence scores
-  **Error Handling**: Robust error handling throughout the codebase
-  **Clean Code**: Well-documented, modular, and maintainable code

## Project Structure

```
AI-Driven-Hashtag-Suggestion-System/
├── LICENSE                              # MIT License
├── README.md                            # Project documentation
├── requirements.txt                     # Python dependencies
├── data/
│   └── ai_hashtag_suggestion_dataset_10k.csv   # Training dataset (20k+ samples)
├── results/
│   ├── hashtag_model.pkl                # Trained model (generated)
│   ├── metrics.txt                      # Evaluation metrics (generated)
│   └── confusion_matrix.png             # Confusion matrix visualization (generated)
└── src/
    ├── train_model.py                   # Model training script
    └── predict.py                       # Prediction/inference script
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AI-Driven-Hashtag-Suggestion-System.git
   cd AI-Driven-Hashtag-Suggestion-System
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   ```
   
   **Activate the virtual environment:**
   ```bash
   # Windows
   .venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

| Package | Version | Purpose |
|---------|---------|----------|
| pandas | 2.3.3 | Data manipulation |
| scikit-learn | 1.8.0 | Machine learning |
| numpy | 2.4.0 | Numerical computing |
| joblib | 1.5.3 | Model serialization |
| matplotlib | 3.10.8 | Visualization |
| seaborn | 0.13.2 | Statistical plots |

## Usage

### Training the Model

Train (or retrain) the model with the dataset:

```bash
python src/train_model.py
```

**Output:**
-  Trained model saved to `results/hashtag_model.pkl`
-  Metrics saved to `results/metrics.txt`
-  Confusion matrix saved to `results/confusion_matrix.png`

### Making Predictions

#### Interactive Mode

Run the prediction script for interactive hashtag suggestions:

```bash
python src/predict.py
```

Example session:
```
==================================================
  AI HASHTAG SUGGESTION SYSTEM
==================================================
Enter a tweet to get hashtag suggestions.
Type 'quit' or 'exit' to stop.

 Model loaded successfully!

 Enter tweet: Just finished an amazing workout at the gym!

  Suggested Hashtags:
------------------------------
   1. #health 🟢 (67.3%)
   2. #motivation 🟡 (18.2%)
   3. #daily_life 🔴 (8.5%)
```

#### Programmatic Usage

```python
from src.predict import suggest_hashtag, suggest_top_hashtags

# Single prediction
hashtag = suggest_hashtag("The stock market is volatile today")
print(f"Suggested: #{hashtag}")  # Output: Suggested: #business

# Top 3 predictions with confidence
tweet = "New iPhone features are incredible"
top_hashtags = suggest_top_hashtags(tweet, top_n=3)
for tag, confidence in top_hashtags:
    print(f"#{tag}: {confidence*100:.1f}%")
```

## Model Performance

| Metric | Score |
|--------|-------|
| **Cross-Validation Accuracy** | 74.76% (±0.79%) |
| **Test Set Accuracy** | 75.38% |
| **Weighted F1-Score** | 75.72% |
| **Macro F1-Score** | 76.31% |


### Per-Category Performance (Top 5)

| Category | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| food | 0.89 | 0.83 | 0.86 |
| fashion | 0.87 | 0.78 | 0.82 |
| science | 0.87 | 0.78 | 0.82 |
| environment | 0.86 | 0.77 | 0.81 |
| health | 0.85 | 0.81 | 0.83 |


### Confusion Matrix

The confusion matrix visualization is automatically generated during training and saved to `results/confusion_matrix.png`.

## Categories

The model classifies tweets into 20 categories:

| Category | Description |
|----------|-------------|
| `business` | Business, finance, startups, economy |
| `culture` | Cultural events, traditions, heritage |
| `daily_life` | Everyday activities, routines |
| `education` | Learning, schools, universities |
| `entertainment` | Movies, TV shows, celebrities |
| `environment` | Climate, sustainability, nature |
| `events` | Conferences, festivals, gatherings |
| `fashion` | Clothing, style, trends |
| `food` | Recipes, restaurants, cooking |
| `gaming` | Video games, esports, gaming culture |
| `health` | Wellness, fitness, mental health |
| `motivation` | Inspirational, self-improvement |
| `music` | Songs, artists, concerts |
| `news` | Current events, breaking news |
| `politics` | Government, policies, elections |
| `science` | Research, discoveries, technology |
| `socialjustice` | Activism, rights, equality |
| `sports` | Athletics, teams, competitions |
| `technology` | Tech, gadgets, software |
| `travel` | Tourism, destinations, adventures |

## How It Works

### Text Preprocessing

1. Convert to lowercase
2. Remove URLs and mentions (@username)
3. Remove special characters and numbers
4. Remove extra whitespace

### Model Pipeline

```
Tweet Text → TF-IDF Vectorization → Logistic Regression → Hashtag Prediction
```

- **TF-IDF Vectorizer**: Converts text to numerical features (max 5000 features, unigrams + bigrams)
- **Logistic Regression**: Multi-class classification with L2 regularization

### Training Process

1. Load and clean dataset
2. Split data (80% train, 20% test) with stratification
3. Perform 5-fold cross-validation
4. Train final model on training set
5. Evaluate on test set
6. Generate confusion matrix
7. Save model and metrics

## API Reference

### `suggest_hashtag(tweet, return_probabilities=False)`

Suggest a hashtag for the given tweet.

**Parameters:**
- `tweet` (str): The tweet text to analyze
- `return_probabilities` (bool): If True, also return confidence scores

**Returns:**
- `str`: Predicted hashtag, or
- `tuple`: (hashtag, probabilities_dict) if return_probabilities=True

### `suggest_top_hashtags(tweet, top_n=3)`

Get top N hashtag suggestions with confidence scores.

**Parameters:**
- `tweet` (str): The tweet text to analyze
- `top_n` (int): Number of top predictions to return

**Returns:**
- `list`: List of tuples (hashtag, probability)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [scikit-learn](https://scikit-learn.org/) team for the excellent ML library
- [pandas](https://pandas.pydata.org/) for data manipulation capabilities
- The open-source community for inspiration and tools

---

<p align="center">
  Made with ❤️ for social media automation
</p>
<p align="center">
  <a href="#">⬆️ Back to Top</a>
</p>
