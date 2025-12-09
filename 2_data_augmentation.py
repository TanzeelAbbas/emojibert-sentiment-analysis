"""
Data Augmentation Script using SMOTE Approximation
This script balances Variant II and III using SMOTE on TF-IDF features,
approximating synthetic texts by nearest neighbors.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import os

def balance_dataset(df, variant_name, output_dir='data/balanced'):
    """
    Balance dataset using SMOTE on TF-IDF features.
    For synthetic samples, assign text from nearest original sample of same class.
    """
    print(f"\nBalancing {variant_name} using SMOTE approximation...")
    
    X = df['text'].values
    y = df['sentiment'].values
    
    # Vectorize text with TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vect = vectorizer.fit_transform(X).toarray()
    
    print(f"Vectorized shape: {X_vect.shape}")
    
    # Prepare per-class indices and NearestNeighbors models
    class_indices = defaultdict(list)
    for idx, label in enumerate(y):
        class_indices[label].append(idx)
    
    nn_models = {}
    for label in class_indices:
        class_X = X_vect[class_indices[label]]
        nn = NearestNeighbors(n_neighbors=1).fit(class_X)
        nn_models[label] = (nn, class_indices[label])
    
    # Apply SMOTE to create synthetic samples
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_vect, y)
    
    print(f"Original shape: {X_vect.shape}")
    print(f"Resampled shape: {X_res.shape}")
    
    # Create list of texts for resampled data
    new_texts = []
    original_len = len(X)
    
    for i in range(len(X_res)):
        if i < original_len:
            # Original sample
            new_texts.append(X[i])
        else:
            # Synthetic sample: find nearest original of same class
            label = y_res[i]
            nn, indices = nn_models[label]
            _, nn_idx = nn.kneighbors([X_res[i]])
            nearest_idx = nn_idx[0][0]
            original_idx = indices[nearest_idx]
            new_texts.append(X[original_idx])
    
    # Create balanced dataframe
    balanced_df = pd.DataFrame({
        'text': new_texts,
        'sentiment': y_res
    })
    
    print(f"Balanced distribution:\n{balanced_df['sentiment'].value_counts()}")
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{variant_name}_balanced.csv"
    balanced_df.to_csv(output_path, index=False)
    print(f"Saved balanced dataset to {output_path}")
    
    return balanced_df

def main():
    print("="*60)
    print("DATA AUGMENTATION WITH SMOTE")
    print("="*60)
    
    # Variants to balance (II and III)
    variants = [
        ('data/variant2_builtin_dict.csv', 'variant2_builtin_dict'),
        ('data/variant3_custom_dict.csv', 'variant3_custom_dict')
    ]
    
    for filepath, variant_name in variants:
        df = pd.read_csv(filepath)
        balance_dataset(df, variant_name)
    
    print("\n" + "="*60)
    print("Data augmentation completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run 3_train_ml_models.py to train ML classifiers")
    print("2. Run 4_train_bert.py to fine-tune BERT model")

if __name__ == "__main__":
    main()