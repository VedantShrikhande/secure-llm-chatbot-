# Secure LLM Chatbot with Content Filtering

**Project title:** Secure LLM Chatbot with Content Filtering

**Objective:** Build a secure chatbot using an LLM that detects and blocks unsafe responses containing PII, hate speech, or biased content. Provide a Streamlit frontend and deliver code, training scripts, logging, and a professional project report.

---

## Repository structure

```
secure-llm-chatbot/
├─ README.md
├─ requirements.txt
├─ app.py                     # Streamlit frontend
├─ backend/
│  ├─ __init__.py
│  ├─ chatbot.py              # LLM wrapper + generation + postprocessing
│  ├─ train_classifier.py     # Train hate-speech classifier
│  ├─ preprocess.py          # dataset preprocessing utilities
├─ filters/
│  ├─ __init__.py
│  ├─ pii_filter.py          # regex PII detection
│  ├─ hate_classifier.py     # ML classifier wrapper (load/predict)
├─ utils/
│  ├─ logger.py              # logging blocked/safe responses
│  ├─ metrics.py             # evaluation utilities
├─ data/
│  ├─ targets.csv            # (example location — your uploaded files)
│  ├─ entries.csv
├─ reports/
│  ├─ project_report.md      # full project report (also PDF-exportable)
```

---

## `requirements.txt`

```
streamlit
transformers
torch
scikit-learn
pandas
numpy
nltk
datasets
joblib
regex
python-dotenv
```

---

## `filters/pii_filter.py`

```python
import re
from typing import List, Tuple

# Patterns for common PII
EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_RE = re.compile(r"(?:\+\d{1,3}[\s-]?)?(?:\d{2,4}[\s-]?){2,4}\d{2,4}")
# Very simple address heuristics (US-centric fragment). Tweak per locale.
ADDRESS_RE = re.compile(r"\d{1,5}\s+\w+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b", re.IGNORECASE)

PII_PATTERNS = [("email", EMAIL_RE), ("phone", PHONE_RE), ("address", ADDRESS_RE)]


def detect_pii(text: str) -> List[Tuple[str, str]]:
    """Return list of (type, match) found in text."""
    found = []
    for name, pat in PII_PATTERNS:
        for m in pat.finditer(text):
            found.append((name, m.group(0)))
    return found


def contains_pii(text: str) -> bool:
    return len(detect_pii(text)) > 0


if __name__ == '__main__':
    sample = "Contact me at john.doe@example.com or call +1 415-555-1234."
    print(detect_pii(sample))
```

---

## `filters/hate_classifier.py`

```python
import joblib
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List

# This module wraps a scikit-learn model saved via joblib.
# Training script (train_classifier.py) saves two artifacts: tfidf_vectorizer.joblib and clf.joblib

ARTIFACT_VECT = 'artifacts/tfidf_vectorizer.joblib'
ARTIFACT_CLF = 'artifacts/clf.joblib'


class HateClassifier:
    def __init__(self, vect_path=ARTIFACT_VECT, clf_path=ARTIFACT_CLF):
        self.vect = joblib.load(vect_path)
        self.clf = joblib.load(clf_path)

    def predict_proba(self, texts: List[str]):
        X = self.vect.transform(texts)
        return self.clf.predict_proba(X)

    def predict(self, texts: List[str]):
        X = self.vect.transform(texts)
        return self.clf.predict(X)


if __name__ == '__main__':
    clf = HateClassifier()
    print(clf.predict(["I love everyone.", "I hate group X."]))
```

---

## `backend/preprocess.py`

```python
import pandas as pd
import re
import nltk
nltk.download('punkt')


def load_and_clean(entries_path: str, targets_path: str):
    """Load uploaded CSVs and join them (assumes parallel indexing)."""
    entries = pd.read_csv(entries_path)
    targets = pd.read_csv(targets_path)

    # Simple check
    if len(entries) != len(targets):
        print('Warning: entry/target length mismatch')

    df = pd.concat([entries, targets], axis=1)

    # Basic cleaning: lowercase, remove excessive whitespace
    df['text'] = df['text'].astype(str).str.strip().str.replace('\n', ' ')
    df['text'] = df['text'].str.replace(r"\s+", ' ', regex=True)

    return df


if __name__ == '__main__':
    df = load_and_clean('data/entries.csv', 'data/targets.csv')
    print(df.head())
```

---

## `backend/train_classifier.py`

```python
"""Train a TF-IDF + LogisticRegression classifier for hate speech detection.
This script expects the user's uploaded CSVs in data/entries.csv and data/targets.csv
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from backend.preprocess import load_and_clean

ARTIFACT_DIR = 'artifacts'


def train(entries_path='data/entries.csv', targets_path='data/targets.csv'):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    df = load_and_clean(entries_path, targets_path)

    # Assume label column is named 'label' with 0 (non-hate) and 1 (hate)
    X = df['text'].fillna('')
    y = df['label'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    vect = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    Xtr = vect.fit_transform(X_train)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, y_train)

    # Save artifacts
    joblib.dump(vect, os.path.join(ARTIFACT_DIR, 'tfidf_vectorizer.joblib'))
    joblib.dump(clf, os.path.join(ARTIFACT_DIR, 'clf.joblib'))

    # Evaluate
    Xte = vect.transform(X_test)
    ypred = clf.predict(Xte)
    print('Accuracy:', accuracy_score(y_test, ypred))
    print(classification_report(y_test, ypred))


if __name__ == '__main__':
    train()
```

---

## `backend/chatbot.py`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from filters.pii_filter import contains_pii, detect_pii
from filters.hate_classifier import HateClassifier
from utils.logger import ConversationLogger

# Choose a small model suitable for local testing; replace with an open LLaMA or larger model if available.
MODEL_NAME = 'distilgpt2'

class SafeChatbot:
    def __init__(self, model_name=MODEL_NAME, device=None):
        self.device = device if device is not None else (0 if torch.cuda.is_available() else -1)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.model.to('cuda')
        self.generator = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer, device=0 if torch.cuda.is_available() else -1)

        # Load hate classifier
        self.hate_clf = HateClassifier()
        self.logger = ConversationLogger('logs/conversations.jsonl')

    def generate_response(self, prompt: str, max_length=128, num_return_sequences=1) -> str:
        # Generate raw candidate
        outputs = self.generator(prompt, max_length=max_length, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=num_return_sequences)
        return outputs[0]['generated_text'][len(prompt):].strip()

    def filter_and_respond(self, user_input: str) -> dict:
        # 1) If user input contains PII, reject immediately (don't echo PII back)
        input_pii = detect_pii(user_input)
        if input_pii:
            blocked = True
            reason = 'PII in user input'
            safe_text = "I'm sorry — I can't process messages that contain personal contact details or other sensitive personal information."
            self.logger.log(user_input, safe_text, blocked, reasons=[reason])
            return {'safe': False, 'text': safe_text, 'blocked': True, 'reason': reason}

        # 2) Generate candidate response from LLM
        candidate = self.generate_response(user_input)

        # 3) Check candidate for PII
        if contains_pii(candidate):
            blocked = True
            reason = 'PII in model output'
            safe_text = "I can't share personal contact details or other personally identifying information."
            self.logger.log(user_input, candidate, blocked, reasons=[reason])
            return {'safe': False, 'text': safe_text, 'blocked': True, 'reason': reason}

        # 4) Check candidate for hate/bias via classifier
        pred = self.hate_clf.predict([candidate])[0]
        if int(pred) == 1:
            blocked = True
            reason = 'Hate/bias content detected'
            safe_text = "I'm here to help, but I can't provide or repeat hateful or abusive content. If you need information on this topic, I can offer neutral, factual context."
            self.logger.log(user_input, candidate, blocked, reasons=[reason])
            return {'safe': False, 'text': safe_text, 'blocked': True, 'reason': reason}

        # 5) Passed all checks — safe to return
        blocked = False
        self.logger.log(user_input, candidate, blocked, reasons=[])
        return {'safe': True, 'text': candidate, 'blocked': False}


if __name__ == '__main__':
    bot = SafeChatbot()
    while True:
        u = input('You: ')
        out = bot.filter_and_respond(u)
        print('Bot:', out['text'])
```

---

## `utils/logger.py`

```python
import json
import os
from datetime import datetime

class ConversationLogger:
    def __init__(self, logfile='logs/conversations.jsonl'):
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        self.logfile = logfile

    def log(self, user_input, model_output, blocked: bool, reasons=None):
        entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'user_input': user_input,
            'model_output': model_output,
            'blocked': bool(blocked),
            'reasons': reasons or []
        }
        with open(self.logfile, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')


if __name__ == '__main__':
    logger = ConversationLogger()
    logger.log('hello', 'hi there', False, [])
```

---

## `app.py` (Streamlit frontend)

```python
import streamlit as st
from backend.chatbot import SafeChatbot

st.set_page_config(page_title='Secure LLM Chatbot', layout='centered')

st.title('🔒 Secure LLM Chatbot')
st.write('A demo chatbot that filters PII and hate/bias content before replying.')

if 'bot' not in st.session_state:
    st.session_state['bot'] = SafeChatbot()

bot: SafeChatbot = st.session_state['bot']

if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

with st.form('input_form', clear_on_submit=True):
    user_input = st.text_area('You:', height=120)
    submitted = st.form_submit_button('Send')

if submitted and use
```
