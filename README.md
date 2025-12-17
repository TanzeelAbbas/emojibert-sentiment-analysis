# Emoji-based Sentiment Analysis for Airline Reviews

Complete implementation of the paper: **"Sentiment analysis of emoji fused reviews using machine learning and BERT"** by Khan, A., Majumdar, D., & Mondal, B. (2025), Scientific Reports.

## 🎯 Project Overview

This project implements a sentiment analysis system that:
- ✅ Incorporates emojis and emoticons as features (not removes them)
- ✅ Uses custom emoji dictionary (EMOJIXT) for sentiment-aware text replacement
- ✅ Implements three variants (no emoji, built-in dict, custom dict)
- ✅ Trains ML classifiers (RF, MNB, SVM, LR) with TF-IDF and Word2Vec
- ✅ Fine-tunes BERT-base model
- ✅ Achieves 94% accuracy (11% improvement over baselines)
- ✅ Applies SMOTE for data balancing

## 📊 Results

| Model | Variant | Accuracy | F1-Score |
|-------|---------|----------|----------|
| **BERT** | **Variant III (Custom Dict)** | **94%** | **0.95** |
| BERT | Variant II (Built-in Dict) | 94% | 0.94 |
| BERT | Variant I (No Emoji) | 85% | 0.79 |
| RF | Variant III | 85% | 0.85 |
| MNB | Variant II | 81% | 0.81 |

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.9+
python --version

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

Create .env file with 
```bash
HF_TOKEN="HuggingFaceTokenGoesHere" 
```

### Step 1: Download Dataset

Download the **Twitter US Airline Sentiment** dataset from Kaggle:
```bash
# Visit: https://www.kaggle.com/crowdflower/twitter-airline-sentiment
# Download Tweets.csv and place it in the project root
```

### Step 2: Data Preparation

```bash
python 1_data_preparation.py
```

This creates three variants:
- `variant1_no_emoji.csv` - Emojis removed
- `variant2_builtin_dict.csv` - Emojis replaced with built-in descriptions
- `variant3_custom_dict.csv` - Emojis replaced with EMOJIXT sentiment words

### Step 3: Data Augmentation

```bash
python 2_data_augmentation.py
```

Applies SMOTE to balance class distribution and reduce overfitting.

### Step 4: Train ML Models

```bash
python 3_train_ml_models.py
```

Trains Random Forest, Multinomial Naive Bayes, SVM, and Logistic Regression on all variants.

### Step 5: Fine-tune BERT

```bash
python 4_train_bert.py
```

Fine-tunes BERT-base on all variants. This takes the longest (20 to 40 mints on GPU, longer on CPU).

**Note:** Training BERT requires significant computational resources. Use Google Colab with GPU for faster training.

### Step 6: Upload to Hugging Face

```bash
# First, login to Hugging Face
huggingface-cli login

# Update HF_USERNAME in the script
nano 5_upload_to_huggingface.py

# Upload model
python 5_upload_to_huggingface.py
```

### Step 7: Create Gradio Space

```bash
# Update HF_USERNAME in the script
nano 6_create_hf_space.py

# Create and deploy Space
python 6_create_hf_space.py
```

## 📁 Project Structure

```
emoji-sentiment-analysis/
├── 1_data_preparation.py          # Data loading and variant creation
├── 2_data_augmentation.py         # SMOTE balancing
├── 3_train_ml_models.py           # ML classifiers training
├── 4_train_bert.py                # BERT fine-tuning
├── 5_upload_to_huggingface.py    # Upload model to HF Hub
├── 6_create_hf_space.py          # Create Gradio Space
├── app.py                         # Gradio interface
├── requirements.txt               # Dependencies
├── README.md                      # This file
├── data/                          # Generated datasets
│   ├── original_data.csv
│   ├── variant1_no_emoji.csv
│   ├── variant2_builtin_dict.csv
│   ├── variant3_custom_dict.csv
│   └── balanced/
│       ├── variant1_no_emoji_balanced.csv
│       ├── variant2_builtin_dict_balanced.csv
│       └── variant3_custom_dict_balanced.csv
└── models/                        # Trained models
    ├── ml/                        # ML classifiers
    │   ├── variant1_tfidf_RandomForest.pkl
    │   ├── variant2_tfidf_MNB.pkl
    │   └── ...
    └── bert/                      # BERT models
        ├── variant1_no_emoji/
        ├── variant2_builtin_dict/
        └── variant3_custom_dict/
```

## 🔬 Implementation Details

### Custom Emoji Dictionary (EMOJIXT)

The EMOJIXT dictionary maps 250 common emojis to sentiment words:

```python
{
    '😊': 'happy joyful',
    '😡': 'furious enraged',
    '✈️': 'airplane flight',
    '👍': 'thumbs up good',
    '😭': 'crying sad',
    # ... 245 more
}
```

### Data Augmentation

- **Technique:** SMOTE (Synthetic Minority Oversampling Technique)
- **Target Ratio:** 0.8 (80% of majority class)
- **Purpose:** Balance dataset from highly imbalanced (63% negative) to more balanced distribution

### BERT Hyperparameters

- **Model:** bert-base-uncased
- **Max Length:** 80 tokens
- **Batch Size:** 16
- **Learning Rate:** 3e-5
- **Epochs:** 10
- **Optimizer:** AdamW
- **Scheduler:** Linear warmup

### ML Feature Extraction

- **TF-IDF:** Max 5000 features, unigrams and bigrams
- **Word2Vec:** 300 dimensions, CBOW and Skip-gram architectures


## 📚 Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{khan2025sentiment,
  title={Sentiment analysis of emoji fused reviews using machine learning and BERT},
  author={Khan, Amit and Majumdar, Dipankar and Mondal, Bikromadittya},
  journal={Scientific Reports},
  volume={15},
  number={1},
  pages={7538},
  year={2025},
  publisher={Nature Publishing Group}
}
```

## 📧 Contact
tanzeelabbas114@gmail.com

---

**Note:** This implementation is for academic and educational purposes. Make sure to update the `HF_USERNAME` in scripts 5 and 6 before deploying to Hugging Face.
