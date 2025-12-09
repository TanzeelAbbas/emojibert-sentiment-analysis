"""
BERT Model Fine-tuning Script
Fine-tunes BERT-base model for sentiment analysis on all variants
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm
import os
import json

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SentimentDataset(Dataset):
    """Custom Dataset for sentiment analysis"""
    
    def __init__(self, texts, labels, tokenizer, max_length=80):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_data(filepath, variant_name):
    """Load dataset"""
    print(f"\nLoading {variant_name}...")
    df = pd.read_csv(filepath)
    print(f"Dataset size: {len(df)}")
    print(f"Class distribution:\n{df['sentiment'].value_counts()}")
    return df

def prepare_dataloaders(df, tokenizer, batch_size=16, max_length=80):
    """Prepare train, validation, and test dataloaders"""
    
    # Split into train, val, test (80-10-10)
    X = df['text'].values
    y = df['sentiment'].values
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1111, random_state=42, stratify=y_temp
    )  # 0.1111 of 0.9 = 0.1 of total
    
    print(f"\nDataset splits:")
    print(f"Train: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Create datasets
    train_dataset = SentimentDataset(X_train, y_train, tokenizer, max_length)
    val_dataset = SentimentDataset(X_val, y_val, tokenizer, max_length)
    test_dataset = SentimentDataset(X_test, y_test, tokenizer, max_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy

def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average=None, labels=[0, 1, 2]
    )
    
    return avg_loss, accuracy, precision, recall, f1, predictions, true_labels

def train_bert_model(df, variant_name, output_dir='models/bert', 
                     epochs=3, batch_size=16, learning_rate=3e-5):
    """Fine-tune BERT model"""
    
    print("\n" + "="*60)
    print(f"Fine-tuning BERT on {variant_name}")
    print("="*60)
    
    # Create output directory
    model_output_dir = f"{output_dir}/{variant_name}"
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Load tokenizer and model
    print("\nLoading BERT-base model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3
    )
    model.to(device)
    
    # Prepare dataloaders
    train_loader, val_loader, test_loader = prepare_dataloaders(
        df, tokenizer, batch_size=batch_size
    )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )
        
        # Validate
        val_loss, val_acc, _, _, _, _, _ = evaluate(model, val_loader, device)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best validation accuracy! Saving model...")
            model.save_pretrained(model_output_dir)
            tokenizer.save_pretrained(model_output_dir)
    
    # Save training history
    with open(f"{model_output_dir}/training_history.json", 'w') as f:
        json.dump(history, f, indent=4)
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    
    # Load best model
    model = BertForSequenceClassification.from_pretrained(model_output_dir)
    model.to(device)
    
    test_loss, test_acc, precision, recall, f1, predictions, true_labels = evaluate(
        model, test_loader, device
    )
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nPer-class metrics:")
    print(f"Negative - Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1: {f1[0]:.4f}")
    print(f"Neutral  - Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1: {f1[1]:.4f}")
    print(f"Positive - Precision: {precision[2]:.4f}, Recall: {recall[2]:.4f}, F1: {f1[2]:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, 
                                target_names=['Negative', 'Neutral', 'Positive']))
    
    # Save results
    results = {
        'variant': variant_name,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'precision_negative': precision[0],
        'precision_neutral': precision[1],
        'precision_positive': precision[2],
        'recall_negative': recall[0],
        'recall_neutral': recall[1],
        'recall_positive': recall[2],
        'f1_negative': f1[0],
        'f1_neutral': f1[1],
        'f1_positive': f1[2],
        'best_val_acc': best_val_acc
    }
    
    return results, history

def main():
    """Main execution function"""
    print("="*60)
    print("BERT MODEL FINE-TUNING")
    print("="*60)
    
    # Define variants
    variants = [
        ('data/variant1_no_emoji.csv', 'variant1_no_emoji'),
        ('data/balanced/variant2_builtin_dict_balanced.csv', 'variant2_builtin_dict'),
        ('data/balanced/variant3_custom_dict_balanced.csv', 'variant3_custom_dict')
    ]
    
    all_results = []
    
    # Train on each variant
    for filepath, variant_name in variants:
        df = load_data(filepath, variant_name)
        
        results, history = train_bert_model(
            df, 
            variant_name,
            epochs= 3,
            batch_size=16,
            learning_rate=3e-5
        )
        
        all_results.append(results)
    
    # Save all results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('results/bert_results.csv', index=False)
    
    print("\n" + "="*60)
    print("BERT TRAINING COMPLETED!")
    print("="*60)
    print("\nResults summary:")
    print(results_df[['variant', 'test_accuracy', 'best_val_acc']].to_string(index=False))

if __name__ == "__main__":
    main()