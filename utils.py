import re
import nltk
import spacy
import torch
from nltk.corpus import stopwords
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Download and load stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load NLP model
nlp = spacy.load("en_core_web_trf")  # Transformer-based SpaCy model

# Load fine-tuned BERT model and tokenizer
def load_model(model_name="aniljoseph/subtheme_sentiment_BERT_finetuned"):
    """Load BERT model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Define unwanted entity types and terms
UNWANTED_ENTITIES = {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}
UNWANTED_TERMS = {"days", "months", "years", "time", "weeks"}

def predict_sentiment(text, tokenizer, model):
    """Predict sentiment using the fine-tuned BERT model."""
    if not text.strip():
        return "neutral"  # Default to neutral if empty

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    # Convert predicted class to label
    labels = ["negative", "positive"]  # Assuming your model follows this order
    return labels[predicted_class]


def clean_text(text):
    """Preprocess text: lowercase, remove special characters, and stopwords."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

nlp = spacy.load("en_core_web_sm")  # Load English NLP model

def extract_meaningful_subthemes(text):
    """Extract meaningful subthemes by combining adjectives with nouns."""
    doc = nlp(text)
    subthemes = {}

    for token in doc:
        # Identify nouns that have adjectives before them
        if token.pos_ == "NOUN":
            # Find adjectives describing the noun
            adj_modifiers = [child.text for child in token.lefts if child.pos_ == "ADJ"]
            if adj_modifiers:
                subtheme = " ".join(adj_modifiers + [token.text])  # Combine adjective + noun
                subthemes[subtheme] = "PRODUCT"
            else:
                subthemes[token.text] = "PRODUCT"

    return subthemes


def analyze_review_detailed(review, tokenizer, model):  # ✅ Add tokenizer and model as parameters
    """Analyze subthemes, extract relevant sentences, and determine sentiment."""
    doc = nlp(review)
    subthemes = extract_meaningful_subthemes(review)
    sentences = list(doc.sents)

    if not subthemes:
        return {"No Subthemes Found": {"sentence": review, "subtheme_category": "N/A", "sentiment": "neutral"}}

    analysis_result = {}
    for subtheme, category in subthemes.items():
        relevant_sentence = next(
            (sent.text for sent in sentences if subtheme.lower() in sent.text.lower()), 
            "Not Found"
        )
        
        # ✅ Pass tokenizer and model correctly to predict_sentiment()
        sentiment = predict_sentiment(relevant_sentence, tokenizer, model) if relevant_sentence != "Not Found" else "neutral"
        
        formatted_subtheme = subtheme.title().replace("The ", "").strip()
        
        if formatted_subtheme not in analysis_result:
            analysis_result[formatted_subtheme] = {
                "sentence": relevant_sentence,
                "subtheme_category": category,
                "sentiment": sentiment
            }
    
    return analysis_result
