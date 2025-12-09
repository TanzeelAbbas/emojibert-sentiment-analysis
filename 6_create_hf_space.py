"""
Script to create and deploy Hugging Face Space
This creates a Gradio app on Hugging Face Spaces
"""

from huggingface_hub import HfApi, create_repo, upload_file
import os
from dotenv import load_dotenv

load_dotenv(override=True)

HF_USERNAME = "tanzeelabbas114"
SPACE_NAME = "emoji-sentiment-demo"
MODEL_REPO = f"{HF_USERNAME}/emojibert-sentiment-analysis"

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("❌ HF_TOKEN environment variable not found. Run: setx HF_TOKEN \"your_key\"")

# Initialize API object with token
api = HfApi(token=HF_TOKEN)

def create_space_readme():
    """Create README.md for the Space"""
    readme = f"""---
title: Airline Sentiment Analysis with Emoji Support
emoji: ✈️😊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: false
license: apache-2.0
---

# Airline Sentiment Analysis with Emoji Support ✈️

An interactive demo for sentiment analysis of airline reviews that incorporates emojis and emoticons for more accurate predictions.

## Model

The model is available at: [{MODEL_REPO}](https://huggingface.co/{MODEL_REPO})
"""
    return readme

def create_app_file():
    """Create app.py with model reference"""
    app_code = f"""import gradio as gr
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import emoji

MODEL_PATH = "{MODEL_REPO}"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

CUSTOM_EMOJI_DICT = {{
    '😊': 'happy joyful', '😃': 'excited cheerful', '😁': 'grinning delighted',
    '😄': 'smiling pleased', '😆': 'laughing amused', '😂': 'hilarious funny',
    '😠': 'angry mad', '😡': 'furious enraged', '😐': 'neutral blank',
    '✈️': 'airplane flight'
}}

def preprocess_text(text):
    for emoji_char, sentiment_words in CUSTOM_EMOJI_DICT.items():
        text = text.replace(emoji_char, f" {{sentiment_words}} ")
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
    sentiment_map = {{0: "Negative 😞", 1: "Neutral 😐", 2: "Positive 😊"}}
    return sentiment_map[predicted_class], {{"Negative": float(probs[0]), "Neutral": float(probs[1]), "Positive": float(probs[2])}}, None

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
"""
    return app_code

# ----------------------------
# Space creation & upload
# ----------------------------

def create_space():
    """Create HuggingFace Space and upload files"""
    repo_id = f"{HF_USERNAME}/{SPACE_NAME}"

    print(f"🚀 Creating HuggingFace Space: {repo_id}")
    create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
        token=HF_TOKEN
    )

    os.makedirs("space_files", exist_ok=True)

    # Create README.md
    with open("space_files/README.md", "w") as f:
        f.write(create_space_readme())

    # Create app.py
    with open("space_files/app.py", "w") as f:
        f.write(create_app_file())

    # Create requirements.txt
    with open("space_files/requirements.txt", "w") as f:
        f.write("gradio\ntransformers\ntorch\nemoji\n")

    # Upload files
    for file in ["README.md", "app.py", "requirements.txt"]:
        upload_file(
            path_or_fileobj=f"space_files/{file}",
            path_in_repo=file,
            repo_id=repo_id,
            repo_type="space",
            token=HF_TOKEN
        )

    print(f"✅ Space deployed: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    create_space()
