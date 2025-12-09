"""
Data Preparation Script for Emoji-based Sentiment Analysis
This script handles data loading, emoji processing, and creates three variants
"""

import pandas as pd
import numpy as np
import re
import emoji
from collections import Counter


def load_data(filepath='Tweets.csv'):
    """Load the Twitter airline sentiment dataset"""
    df = pd.read_csv(filepath)
    
    # Select relevant columns
    df = df[['text', 'airline_sentiment']].copy()
    df.columns = ['text', 'sentiment']
    
    # Map sentiment labels
    sentiment_map = {'positive': 2, 'neutral': 1, 'negative': 0}
    df['sentiment'] = df['sentiment'].map(sentiment_map)
    
    # Remove any NaN values
    df = df.dropna()
    
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
    
    return df

def extract_emojis(text):
    """Extract all emojis from text"""
    return ''.join(c for c in text if c in emoji.EMOJI_DATA)

def has_emoji(text):
    """Check if text contains emojis"""
    return bool(extract_emojis(text))

def create_custom_emoji_dict():
    """
    Create custom emoji dictionary (EMOJIXT) with sentiment-based mappings
    This is a simplified version - expand based on your needs
    """
    emoji_dict = {
        '😊': 'happy joyful',
        '😃': 'excited cheerful',
        '😁': 'grinning delighted',
        '😄': 'smiling pleased',
        '😆': 'laughing amused',
        '😂': 'hilarious funny',
        '🤣': 'rofl laughing',
        '😍': 'love adore',
        '🥰': 'loving affectionate',
        '😘': 'kiss loving',
        '😗': 'kissing sweet',
        '☺️': 'content happy',
        '😇': 'angelic innocent',
        '🙂': 'slightly happy',
        '🙃': 'upside down silly',
        '😉': 'wink playful',
        '😌': 'relieved peaceful',
        '😔': 'sad disappointed',
        '😞': 'unhappy sorrowful',
        '😟': 'worried concerned',
        '😠': 'angry mad',
        '😡': 'furious enraged',
        '🤬': 'cursing angry',
        '😤': 'frustrated annoyed',
        '😭': 'crying sad',
        '😢': 'tearful upset',
        '😥': 'disappointed sad',
        '😰': 'anxious nervous',
        '😨': 'fearful scared',
        '😱': 'shocked terrified',
        '😖': 'confounded frustrated',
        '😣': 'persevering struggling',
        '😓': 'downcast sad',
        '😩': 'weary tired',
        '😫': 'exhausted drained',
        '🥱': 'yawning tired',
        '😴': 'sleeping tired',
        '😪': 'sleepy drowsy',
        '🤔': 'thinking pondering',
        '🤨': 'skeptical questioning',
        '😐': 'neutral blank',
        '😑': 'expressionless meh',
        '😶': 'speechless silent',
        '🙄': 'eyeroll annoyed',
        '😏': 'smirking sly',
        '😒': 'unamused bored',
        '🙁': 'frowning sad',
        '☹️': 'sad unhappy',
        '😕': 'confused puzzled',
        '😬': 'grimacing awkward',
        '🤐': 'silent quiet',
        '🤢': 'nauseated sick',
        '🤮': 'vomiting disgusted',
        '🤧': 'sneezing sick',
        '😷': 'masked sick',
        '🤒': 'ill sick',
        '🤕': 'injured hurt',
        '👍': 'thumbs up good',
        '👎': 'thumbs down bad',
        '👏': 'clapping applause',
        '🙏': 'praying thankful',
        '❤️': 'heart love',
        '💔': 'heartbreak sad',
        '💯': 'hundred perfect',
        '✈️': 'airplane flight',
        '🛫': 'departure takeoff',
        '🛬': 'arrival landing',
        '⭐': 'star excellent',
        '🌟': 'glowing star amazing',
        '✨': 'sparkles wonderful',
        '🎉': 'celebration party',
        '🎊': 'confetti celebration',
        '🔥': 'fire hot amazing',
        '💪': 'strong powerful',
        '👌': 'okay good',
        '✅': 'checkmark correct',
        '❌': 'cross wrong',
        '⚠️': 'warning caution',
        '🚫': 'prohibited bad',
        '💩': 'poop bad terrible',
        '🤷': 'shrug whatever',
        '😎': 'cool awesome',
    }
    return emoji_dict

def variant1_remove_emojis(df):
    """Variant I: Remove all emojis and emoticons"""
    df_variant1 = df.copy()
    
    def remove_emoji(text):
        # Remove emojis
        text = emoji.replace_emoji(text, replace='')
        # Remove common emoticons
        emoticon_pattern = r'[:;=8][\-o\*\']?[\)\]\(\[dDpP/\:\}\{@\|\\]'
        text = re.sub(emoticon_pattern, '', text)
        # Clean extra spaces
        text = ' '.join(text.split())
        return text
    
    df_variant1['text'] = df_variant1['text'].apply(remove_emoji)
    
    print(f"\nVariant I created: {len(df_variant1)} samples (emojis removed)")
    return df_variant1

def variant2_builtin_dict(df):
    """Variant II: Replace emojis with built-in emoji dictionary"""
    df_variant2 = df.copy()
    
    def replace_emoji_builtin(text):
        # Use emoji library to replace emojis with their description
        text = emoji.demojize(text, delimiters=(" ", " "))
        # Remove underscores from emoji descriptions
        text = text.replace('_', ' ')
        # Remove common emoticons
        emoticon_pattern = r'[:;=8][\-o\*\']?[\)\]\(\[dDpP/\:\}\{@\|\\]'
        emoticon_replacements = {
            ':)': 'happy', ':(': 'sad', ':D': 'very happy',
            ';)': 'wink', ':P': 'playful', ':O': 'surprised',
            ':|': 'neutral', ':/': 'confused', '<3': 'love'
        }
        for emoticon, replacement in emoticon_replacements.items():
            text = text.replace(emoticon, replacement)
        text = re.sub(emoticon_pattern, '', text)
        # Clean extra spaces
        text = ' '.join(text.split())
        return text
    
    df_variant2['text'] = df_variant2['text'].apply(replace_emoji_builtin)
    
    print(f"Variant II created: {len(df_variant2)} samples (emojis replaced with built-in dict)")
    return df_variant2

def variant3_custom_dict(df):
    """Variant III: Replace emojis with custom EMOJIXT dictionary"""
    df_variant3 = df.copy()
    custom_dict = create_custom_emoji_dict()
    
    def replace_emoji_custom(text):
        # Replace emojis with custom sentiment words
        for emoji_char, sentiment_words in custom_dict.items():
            if emoji_char in text:
                text = text.replace(emoji_char, f" {sentiment_words} ")
        
        # Handle emoticons
        emoticon_replacements = {
            ':)': 'happy positive', ':(': 'sad negative', ':D': 'very happy excited',
            ';)': 'wink playful', ':P': 'playful fun', ':O': 'surprised shocked',
            ':|': 'neutral indifferent', ':/': 'confused uncertain', '<3': 'love affection'
        }
        for emoticon, replacement in emoticon_replacements.items():
            text = text.replace(emoticon, replacement)
        
        # Remove any remaining emojis not in dictionary
        text = emoji.replace_emoji(text, replace='')
        
        # Clean extra spaces
        text = ' '.join(text.split())
        return text
    
    df_variant3['text'] = df_variant3['text'].apply(replace_emoji_custom)
    
    print(f"Variant III created: {len(df_variant3)} samples (emojis replaced with EMOJIXT)")
    return df_variant3

def analyze_dataset(df):
    """Analyze dataset for emoji statistics"""
    df['has_emoji'] = df['text'].apply(has_emoji)
    df['emoji_count'] = df['text'].apply(lambda x: len(extract_emojis(x)))
    
    print(f"\nDataset Analysis:")
    print(f"Total tweets: {len(df)}")
    print(f"Tweets with emojis: {df['has_emoji'].sum()} ({df['has_emoji'].sum()/len(df)*100:.2f}%)")
    print(f"Tweets without emojis: {(~df['has_emoji']).sum()} ({(~df['has_emoji']).sum()/len(df)*100:.2f}%)")
    
    return df

def save_variants(df, df_v1, df_v2, df_v3, output_dir='data'):
    """Save all variants to CSV files"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    df.to_csv(f'{output_dir}/original_data.csv', index=False)
    df_v1.to_csv(f'{output_dir}/variant1_no_emoji.csv', index=False)
    df_v2.to_csv(f'{output_dir}/variant2_builtin_dict.csv', index=False)
    df_v3.to_csv(f'{output_dir}/variant3_custom_dict.csv', index=False)
    
    print(f"\nAll variants saved to '{output_dir}' directory")

def main():
    """Main execution function"""
    print("="*60)
    print("EMOJI-BASED SENTIMENT ANALYSIS - DATA PREPARATION")
    print("="*60)
    
    # Load data
    df = load_data('Tweets.csv')
    
    # Analyze dataset
    df = analyze_dataset(df)
    
    # Create variants
    print("\n" + "="*60)
    print("Creating Variants...")
    print("="*60)
    
    df_variant1 = variant1_remove_emojis(df)
    df_variant2 = variant2_builtin_dict(df)
    df_variant3 = variant3_custom_dict(df)
    
    # Save all variants
    save_variants(df, df_variant1, df_variant2, df_variant3)
    
    print("\n" + "="*60)
    print("Data preparation completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run data_augmentation.py to balance the dataset")
    print("2. Run train_ml_models.py to train ML classifiers")
    print("3. Run train_bert.py to fine-tune BERT model")

if __name__ == "__main__":
    main()