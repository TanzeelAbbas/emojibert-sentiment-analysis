import gradio as gr
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import emoji
import re

# Load model and tokenizer
MODEL_PATH = "tanzeelabbas114/emojibert-sentiment-analysis"  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading model...")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()
print("Model loaded successfully!")

# Custom emoji dictionary (EMOJIXT)
CUSTOM_EMOJI_DICT = {
    '😊': 'happy joyful', '😃': 'excited cheerful', '😁': 'grinning delighted',
    '😄': 'smiling pleased', '😆': 'laughing amused', '😂': 'hilarious funny',
    '🤣': 'rofl laughing', '😍': 'love adore', '🥰': 'loving affectionate',
    '😘': 'kiss loving', '😗': 'kissing sweet', '☺️': 'content happy',
    '😇': 'angelic innocent', '🙂': 'slightly happy', '😉': 'wink playful',
    '😌': 'relieved peaceful', '😔': 'sad disappointed', '😞': 'unhappy sorrowful',
    '😟': 'worried concerned', '😠': 'angry mad', '😡': 'furious enraged',
    '🤬': 'cursing angry', '😤': 'frustrated annoyed', '😭': 'crying sad',
    '😢': 'tearful upset', '😥': 'disappointed sad', '😰': 'anxious nervous',
    '😨': 'fearful scared', '😱': 'shocked terrified', '😖': 'confounded frustrated',
    '😣': 'persevering struggling', '😓': 'downcast sad', '😩': 'weary tired',
    '😫': 'exhausted drained', '🥱': 'yawning tired', '😴': 'sleeping tired',
    '😪': 'sleepy drowsy', '🤔': 'thinking pondering', '😐': 'neutral blank',
    '😑': 'expressionless meh', '😶': 'speechless silent', '🙄': 'eyeroll annoyed',
    '😏': 'smirking sly', '😒': 'unamused bored', '🙁': 'frowning sad',
    '☹️': 'sad unhappy', '😕': 'confused puzzled', '😬': 'grimacing awkward',
    '👍': 'thumbs up good', '👎': 'thumbs down bad', '👏': 'clapping applause',
    '🙏': 'praying thankful', '❤️': 'heart love', '💔': 'heartbreak sad',
    '💯': 'hundred perfect', '✈️': 'airplane flight', '🛫': 'departure takeoff',
    '🛬': 'arrival landing', '⭐': 'star excellent', '🌟': 'glowing star amazing',
    '✨': 'sparkles wonderful', '🎉': 'celebration party', '🔥': 'fire hot amazing',
    '💪': 'strong powerful', '👌': 'okay good', '✅': 'checkmark correct',
    '❌': 'cross wrong', '⚠️': 'warning caution', '🚫': 'prohibited bad',
    '💩': 'poop bad terrible', '😎': 'cool awesome',
}

def preprocess_text(text, use_emoji_replacement=True):
    """Preprocess text with emoji handling"""
    if use_emoji_replacement:
        # Replace emojis with sentiment words
        for emoji_char, sentiment_words in CUSTOM_EMOJI_DICT.items():
            if emoji_char in text:
                text = text.replace(emoji_char, f" {sentiment_words} ")
        
        # Handle common emoticons
        emoticon_map = {
            ':)': 'happy positive', ':(': 'sad negative', ':D': 'very happy excited',
            ';)': 'wink playful', ':P': 'playful fun', ':O': 'surprised shocked',
            ':|': 'neutral indifferent', ':/': 'confused uncertain', '<3': 'love affection'
        }
        for emoticon, replacement in emoticon_map.items():
            text = text.replace(emoticon, replacement)
        
        # Remove any remaining emojis
        text = emoji.replace_emoji(text, replace='')
    
    # Clean text
    text = ' '.join(text.split())
    return text

def predict_sentiment(text, show_preprocessing=False):
    """Predict sentiment for input text"""
    
    if not text.strip():
        return "Please enter some text!", None, None
    
    # Preprocess
    processed_text = preprocess_text(text, use_emoji_replacement=True)
    
    # Tokenize
    inputs = tokenizer.encode_plus(
        processed_text,
        add_special_tokens=True,
        max_length=80,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Predict
    with torch.no_grad():
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
    
    # Map predictions
    sentiment_map = {0: "Negative 😞", 1: "Neutral 😐", 2: "Positive 😊"}
    sentiment_label = sentiment_map[predicted_class]
    
    # Create probability distribution
    prob_dict = {
        "Negative": float(probabilities[0]),
        "Neutral": float(probabilities[1]),
        "Positive": float(probabilities[2])
    }
    
    # Preprocessing info
    preprocessing_info = f"**Original Text:** {text}\n\n**Processed Text:** {processed_text}" if show_preprocessing else None
    
    return sentiment_label, prob_dict, preprocessing_info

# Example texts
examples = [
    "I love this airline! Best experience ever! ✈️😊",
    "Terrible service. My flight was delayed for 5 hours 😡",
    "The flight was okay, nothing special 😐",
    "Amazing crew and comfortable seats! Would definitely fly again 🌟👍",
    "Worst airline ever! Lost my luggage and rude staff 😠💔",
    "Flight cancelled without proper notice 😤",
    "Great food and entertainment options ✨",
    "Neutral experience, met expectations",
]

# Create Gradio interface
with gr.Blocks(title="Emoji-based Sentiment Analysis", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown(
        """
        # Sentiment Analysis with Emoji Support
        
        This model analyzes reviews and incorporates **emojis and emoticons** for more accurate sentiment detection.
        
        ### 📊 Paper Implementation
        Based on: *"Sentiment analysis of emoji fused reviews using machine learning and BERT"*
        
        **Model:** BERT-base fine-tuned on US Airline Twitter dataset with custom emoji dictionary (EMOJIXT)
        
        **Accuracy:** 94% (11% improvement over baseline models)
        """
    )
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Enter your review",
                placeholder="Type or paste your review here... Feel free to use emojis! 😊",
                lines=4
            )
            
            show_preprocessing = gr.Checkbox(
                label="Show text preprocessing steps",
                value=False
            )
            
            predict_btn = gr.Button("Analyze Sentiment 🔍", variant="primary", size="lg")
        
        with gr.Column():
            sentiment_output = gr.Textbox(
                label="Predicted Sentiment",
                interactive=False
            )
            
            probability_output = gr.Label(
                label="Confidence Scores",
                num_top_classes=3
            )
    
    preprocessing_output = gr.Markdown(label="Preprocessing Details", visible=True)
    
    # Examples section
    gr.Markdown("### 📝 Try these examples:")
    gr.Examples(
        examples=examples,
        inputs=text_input,
        outputs=[sentiment_output, probability_output, preprocessing_output],
        fn=predict_sentiment,
        cache_examples=False
    )
    
    # About section
    with gr.Accordion("ℹ️ About this Model", open=False):
        gr.Markdown(
            """
            ### Key Features:
            - **Emoji Integration:** Uses custom EMOJIXT dictionary to convert emojis into sentiment-bearing text
            - **BERT-base Architecture:** 12 transformer layers, 768 hidden units
            - **Multi-class Classification:** Negative, Neutral, Positive sentiments
            - **Data Augmentation:** SMOTE technique used to balance training data
            
            ### Performance Metrics:
            - **Accuracy:** 94%
            - **Precision (Positive):** 0.95
            - **Recall (Positive):** 0.94
            - **F1-Score (Positive):** 0.95
            
            ### Dataset:
            - US Airline Twitter Sentiment Dataset
            - 14,640 reviews total
            - ~6% contain emojis/emoticons
            
            ### Citation:
            ```
            Khan, A., Majumdar, D., & Mondal, B. (2025). 
            Sentiment analysis of emoji fused reviews using machine learning and BERT. 
            Scientific Reports, 15(1), 7538.
            ```
            """
        )
    
    # Connect button to function
    predict_btn.click(
        fn=predict_sentiment,
        inputs=[text_input, show_preprocessing],
        outputs=[sentiment_output, probability_output, preprocessing_output]
    )

# Launch app
if __name__ == "__main__":
    demo.launch()
