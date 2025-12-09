"""
ML Models Training Script
Trains multiple ML classifiers on all variants using TF-IDF and Word2Vec features.
Note: Word2Vec requires gensim and internet for first download of 'word2vec-google-news-300'.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)
import gensim.downloader as api
from tqdm import tqdm
import os
import json

# Load pre-trained Word2Vec 
print("Loading Word2Vec model...")
word2vec = api.load('word2vec-google-news-300')

def get_sentence_embedding(text, model):
    """Compute average Word2Vec embedding for sentence"""
    words = text.lower().split()
    vectors = [model[word] for word in words if word in model]
    if not vectors:
        return np.zeros(300)
    return np.mean(vectors, axis=0)

def prepare_features(train_df, test_df, feature_type='tfidf'):
    """Prepare features based on type"""
    if feature_type == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train = vectorizer.fit_transform(train_df['text']).toarray()
        X_test = vectorizer.transform(test_df['text']).toarray()
    elif feature_type == 'word2vec':
        X_train = np.array([get_sentence_embedding(text, word2vec) for text in tqdm(train_df['text'], desc="Embedding train")])
        X_test = np.array([get_sentence_embedding(text, word2vec) for text in tqdm(test_df['text'], desc="Embedding test")])
    else:
        raise ValueError("Invalid feature_type")
    
    y_train = train_df['sentiment'].values
    y_test = test_df['sentiment'].values
    
    return X_train, X_test, y_train, y_test

def train_evaluate_classifier(clf, X_train, y_train, X_test, y_test):
    """Train and evaluate a classifier"""
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average=None, labels=[0,1,2]
    )
    
    return accuracy, precision, recall, f1, predictions

def train_on_variant(df, variant_name, feature_type, output_dir='models/ml'):
    """Train ML models on a variant with specific features"""
    print("\n" + "="*60)
    print(f"Training on {variant_name} with {feature_type.upper()}")
    print("="*60)
    
    # Split into train/test
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['sentiment']
    )
    
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    
    # Prepare features
    X_train, X_test, y_train, y_test = prepare_features(
        train_df, test_df, feature_type
    )
    
    # Define classifiers (MultinomialNB only for TF-IDF as it requires non-negative features)
    classifiers = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'MultinomialNB': MultinomialNB() if feature_type == 'tfidf' else None,
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'SVC': SVC(random_state=42)
    }
    
    results = {}
    
    for clf_name, clf in classifiers.items():
        if clf is None:
            continue
        print(f"\nTraining {clf_name}...")
        acc, prec, rec, f1, preds = train_evaluate_classifier(
            clf, X_train, y_train, X_test, y_test
        )
        
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, preds, target_names=['Negative', 'Neutral', 'Positive']))
        
        results[clf_name] = {
            'accuracy': acc,
            'precision': prec.tolist(),
            'recall': rec.tolist(),
            'f1': f1.tolist()
        }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{variant_name}_{feature_type}_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {output_path}")
    
    return results

def main():
    print("="*60)
    print("ML MODELS TRAINING")
    print("="*60)
    
    # Define variants (variant1 unbalanced, others balanced)
    variants = [
        ('data/variant1_no_emoji.csv', 'variant1_no_emoji'),
        ('data/balanced/variant2_builtin_dict_balanced.csv', 'variant2_builtin_dict'),
        ('data/balanced/variant3_custom_dict_balanced.csv', 'variant3_custom_dict')
    ]
    
    feature_types = ['tfidf', 'word2vec']
    
    all_results = []
    
    for filepath, variant_name in variants:
        df = pd.read_csv(filepath)
        for feat_type in feature_types:
            results = train_on_variant(df, variant_name, feat_type)
            all_results.append({
                'variant': variant_name,
                'features': feat_type,
                'results': results
            })
    
    # Create and save summary dataframe
    summary_data = []
    for r in all_results:
        row = {
            'variant': r['variant'],
            'features': r['features'],
            'rf_acc': r['results'].get('RandomForest', {}).get('accuracy'),
            'mnb_acc': r['results'].get('MultinomialNB', {}).get('accuracy'),
            'lr_acc': r['results'].get('LogisticRegression', {}).get('accuracy'),
            'svc_acc': r['results'].get('SVC', {}).get('accuracy')
        }
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('models/ml/ml_summary.csv', index=False)
    
    print("\n" + "="*60)
    print("ML TRAINING COMPLETED!")
    print("="*60)
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    
    print("\nNext step:")
    print("Run 4_train_bert.py to fine-tune BERT model")

if __name__ == "__main__":
    main()