# Import necessary libraries
import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib

# Paths to required files (Update these paths accordingly)
MODEL_DIR = r'C:\Training codes\jupyter\News_category_classification'
LABEL_ENCODER_PATH = r'C:\Training codes\jupyter\News_category_classification\label_encoder.pkl'

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="News Headline Categorization",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
st.markdown(
    """
    <style>
    /* Customize title and subtitle */
    .title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: #2c3e50;
    }
    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #7f8c8d;
    }
    /* Center align input box and button */
    .stTextInput {
        margin: 0 auto;
        text-align: center;
    }
    /* Enhance prediction result appearance */
    .result-box {
        background-color: #ecf0f1;
        padding: 20px;
        border-radius: 8px;
        font-size: 18px;
        text-align: center;
        border: 1px solid #bdc3c7;
        color: #2c3e50;
    }
    /* Footer */
    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=13)  # Adjust num_labels as needed
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    return model, tokenizer

# Function to load label encoder
@st.cache_resource
def load_label_encoder():
    return joblib.load(LABEL_ENCODER_PATH)

# Prediction function
def predict_category(headline, model, tokenizer, label_encoder):
    """
    Predict the category for a given headline.

    Args:
        headline (str): Input headline.
        model (BertForSequenceClassification): The trained BERT model.
        tokenizer (BertTokenizer): Tokenizer for text preprocessing.
        label_encoder (LabelEncoder): Fitted label encoder for decoding labels.

    Returns:
        str: Predicted category label.
    """
    inputs = tokenizer(
        headline, 
        padding=True, 
        truncation=True, 
        max_length=128, 
        return_tensors="pt"
    )
    inputs = {key: val.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    predicted_label = label_encoder.inverse_transform([predicted_class_id])
    return predicted_label[0]

# Load model, tokenizer, and label encoder
model, tokenizer = load_model_and_tokenizer()
label_encoder = load_label_encoder()

# Streamlit App UI
st.markdown("<div class='title'>📰 News Headline Categorization</div>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="subtitle">
        Classify news headlines into predefined categories with the power of a fine-tuned BERT model.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# Input Section
st.subheader("Enter a News Headline:")
headline = st.text_input(
    "Type your headline below and press 'Classify':",
    placeholder="E.g., Scientists discover new planet outside solar system."
)

# Prediction Section
if st.button("Classify"):
    st.markdown("---")
    if headline.strip():
        # Show loading spinner during prediction
        with st.spinner("Classifying the headline..."):
            category = predict_category(headline, model, tokenizer, label_encoder)
        st.markdown(
            f"<div class='result-box'>Predicted Category: <b>{category}</b></div>",
            unsafe_allow_html=True
        )
    else:
        st.error("❗ Please enter a valid headline!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #95a5a6;">
        Made with ❤️ using <a href="https://streamlit.io/" target="_blank" style="color: #3498db;">Streamlit</a> and <a href="https://huggingface.co/transformers/" target="_blank" style="color: #3498db;">HuggingFace Transformers</a>.
    </div>
    """,
    unsafe_allow_html=True
)
