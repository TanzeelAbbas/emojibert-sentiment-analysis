import gradio as gr
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import emoji

MODEL_PATH = "tanzeelabbas114/emojibert-sentiment-analysis"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

CUSTOM_EMOJI_DICT = {
    '😊': 'happy joyful', '😃': 'excited cheerful', '😁': 'grinning delighted',
    '😄': 'smiling pleased', '😆': 'laughing amused', '😂': 'hilarious funny',
    '😠': 'angry mad', '😡': 'furious enraged', '😐': 'neutral blank',
    '✈️': 'airplane flight'
}

def preprocess_text(text):
    for emoji_char, sentiment_words in CUSTOM_EMOJI_DICT.items():
        text = text.replace(emoji_char, f" {sentiment_words} ")
    text = emoji.replace_emoji(text, replace='')
    text = ' '.join(text.split())
    return text

def predict_sentiment(text, show_preprocessing=False):
    if not text.strip():
        return "Please enter some text!", None, None
    processed_text = preprocess_text(text)
    inputs = tokenizer.encode_plus(processed_text, add_special_tokens=True,
                                   max_length=80, padding='max_length',
                                   truncation=True, return_attention_mask=True,
                                   return_tensors='pt')
    with torch.no_grad():
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        predicted_class = torch.argmax(probs).item()
    sentiment_map = {0: "Negative 😞", 1: "Neutral 😐", 2: "Positive 😊"}
    return sentiment_map[predicted_class], {"Negative": float(probs[0]), "Neutral": float(probs[1]), "Positive": float(probs[2])}, None

examples = [
    "I love this airline! Best experience ever! ✈️😊",
    "Terrible service. My flight was delayed for 5 hours 😡"
]

with gr.Blocks() as demo:
    text_input = gr.Textbox(label="Enter your review")
    sentiment_output = gr.Textbox(label="Predicted Sentiment")
    predict_btn = gr.Button("Analyze")
    predict_btn.click(fn=predict_sentiment, inputs=text_input, outputs=sentiment_output)

demo.launch()
