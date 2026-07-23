"""
Emotion Classification — Gradio App (standalone, run locally)

Usage:
    pip install -r requirements.txt
    python app.py

Then open the local URL Gradio prints (usually http://127.0.0.1:7860).

Expects a fine-tuned DistilBERT sequence-classification model saved with
`model.save_pretrained(...)` / `tokenizer.save_pretrained(...)` at MODEL_DIR
below (e.g. copy the `distilbert_emotion` folder you exported from Colab/Kaggle
next to this file).
"""

import os
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
MODEL_DIR = "."
TARGET_EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]
NUM_CLASSES = len(TARGET_EMOTIONS)

EMOTION_EMOJI = {
    "joy": "\U0001F604", "surprise": "\U0001F62E", "fear": "\U0001F628",
    "sadness": "\U0001F61F", "anger": "\U0001F620", "disgust": "\U0001F922",
}
# Order the confidence list is displayed in
DISPLAY_ORDER = ["joy", "surprise", "fear", "sadness", "anger", "disgust"]

# ----------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------
if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(
        f"Model folder not found at '{MODEL_DIR}'.\n"
        f"Set the MODEL_DIR environment variable or edit MODEL_DIR at the top "
        f"of app.py to point at your exported DistilBERT folder "
        f"(the one containing config.json, model.safetensors, tokenizer files)."
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# ----------------------------------------------------------------------
# Theme (same dark navy / coral gradient look as the reference design)
# ----------------------------------------------------------------------
CUSTOM_CSS = """
.gradio-container {
    background: linear-gradient(180deg, #1a0e2e 0%, #6e2140 18%, #c9524f 36%,
                #e6835e 50%, #eaa679 62%, #2b3352 80%, #141829 100%) !important;
    font-family: 'Segoe UI', -apple-system, sans-serif !important;
}
#app-header h1 {
    color: #ff8a75 !important;
    font-weight: 800 !important;
    text-align: center;
    font-size: 2.3em !important;
}
#app-header p {
    color: #eef0f5 !important;
    text-align: center;
    font-size: 1.05em !important;
    opacity: 0.9;
}
.main-card {
    background: #141a2e !important;
    border: 1px solid rgba(255,130,110,0.25) !important;
    border-radius: 18px !important;
    padding: 24px !important;
}
label span, .gr-label, .block label {
    color: #ffb3a1 !important;
    font-weight: 600 !important;
}
textarea, input[type="text"] {
    background: #101527 !important;
    color: #f0f0f5 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
}
#predict-btn {
    background: linear-gradient(135deg, #ff6f5e, #e8455a) !important;
    color: white !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
}
#clear-btn {
    background: #241b2f !important;
    color: #e8e8ef !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
}
.result-card {
    background: #101527;
    border: 1px solid rgba(255,130,110,0.35);
    border-radius: 14px;
    padding: 22px;
    text-align: center;
}
.result-card .emoji { font-size: 3.2em; }
.result-card .label { color: #ff8a75; font-size: 1.6em; font-weight: 800; margin-top: 6px; }
.bars-wrap { display: flex; flex-direction: column; gap: 10px; }
.bar-row { display: flex; align-items: center; gap: 10px; }
.bar-row .emoji { font-size: 1.3em; width: 28px; }
.bar-row .name { color: #eef0f5; width: 80px; font-weight: 600; }
.bar-track { flex: 1; background: #1f2540; border-radius: 8px; height: 18px; overflow: hidden; }
.bar-fill { height: 100%; background: linear-gradient(90deg, #ff6f5e, #ff9c72); border-radius: 8px; }
.bar-pct { color: #eef0f5; width: 42px; text-align: right; font-weight: 600; }
.footer-note { text-align: center; color: #cfd3de; opacity: 0.7; margin-top: 10px; }
"""

EMPTY_RESULT_HTML = "<div class='result-card'><div class='emoji'>\U0001F914</div><div class='label'>\u2014</div></div>"


def render_result_card(top_emotion: str) -> str:
    return f"""
    <div class="result-card">
      <div class="emoji">{EMOTION_EMOJI[top_emotion]}</div>
      <div class="label">{top_emotion.capitalize()}</div>
    </div>
    """


def render_bars(probs_dict: dict) -> str:
    rows = ""
    for emo in DISPLAY_ORDER:
        pct = probs_dict[emo] * 100
        rows += f"""
        <div class="bar-row">
          <span class="emoji">{EMOTION_EMOJI[emo]}</span>
          <span class="name">{emo.capitalize()}</span>
          <div class="bar-track"><div class="bar-fill" style="width:{pct:.0f}%"></div></div>
          <span class="bar-pct">{pct:.0f}%</span>
        </div>
        """
    return f'<div class="bars-wrap">{rows}</div>'


def predict_emotion(text: str):
    if not text or not text.strip():
        return EMPTY_RESULT_HTML, render_bars({e: 0.0 for e in DISPLAY_ORDER})

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).numpy()[0]
    probs_dict = {TARGET_EMOTIONS[i]: float(probs[i]) for i in range(NUM_CLASSES)}

    top_emotion = max(probs_dict, key=probs_dict.get)
    return render_result_card(top_emotion), render_bars(probs_dict)


def clear_all():
    return "", EMPTY_RESULT_HTML, render_bars({e: 0.0 for e in DISPLAY_ORDER})


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
with gr.Blocks(css=CUSTOM_CSS, title="Emotion Classification") as demo:
    with gr.Column(elem_id="app-header"):
        gr.Markdown("# \U0001F9E0 Emotion Classification")
        gr.Markdown("DistilBERT + Attention Model")

    with gr.Column(elem_classes="main-card"):
        text_in = gr.Textbox(
            label="Enter your text", lines=3,
            placeholder="I am so excited to graduate today!",
        )
        with gr.Row():
            predict_btn = gr.Button("\u2728 Predict Emotion", elem_id="predict-btn")
            clear_btn = gr.Button("\U0001F5D1\uFE0F Clear", elem_id="clear-btn")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**Predicted Emotion**")
                result_html = gr.HTML(EMPTY_RESULT_HTML)
            with gr.Column(scale=1):
                gr.Markdown("**Confidence Scores**")
                bars_html = gr.HTML(render_bars({e: 0.0 for e in DISPLAY_ORDER}))

        gr.Markdown(
            "<div class='footer-note'>Model: DistilBERT (fine-tuned) &nbsp;|&nbsp; "
            "6 Classes: Joy, Sadness, Anger, Fear, Surprise, Disgust &nbsp;|&nbsp; "
            "Built with Gradio \u2022 Transformers \u2022 PyTorch</div>"
        )

        gr.Examples(
            examples=[
                "I am so excited to graduate today!",
                "I can't believe you did this to me, I'm furious!",
                "I'm so scared about the exam results tomorrow.",
                "What a wonderful, unexpected surprise!",
                "That smell is absolutely disgusting.",
                "I miss her so much, it hurts every day.",
            ],
            inputs=text_in,
        )

    predict_btn.click(fn=predict_emotion, inputs=text_in, outputs=[result_html, bars_html])
    clear_btn.click(fn=clear_all, inputs=None, outputs=[text_in, result_html, bars_html])
    text_in.submit(fn=predict_emotion, inputs=text_in, outputs=[result_html, bars_html])


if __name__ == "__main__":
    demo.launch()
