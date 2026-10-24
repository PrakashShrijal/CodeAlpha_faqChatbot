# FAQ Chatbot – NLP-Powered Customer Support Bot

A complete, fully local FAQ chatbot that answers customer questions using **NLP text preprocessing**, **TF-IDF vectorization**, and **cosine similarity**.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Project Structure](#project-structure)
4. [Setup and Installation](#setup-and-installation)
5. [Running the Application](#running-the-application)
6. [Accessing the Chatbot](#accessing-the-chatbot)
7. [Example Questions to Test](#example-questions-to-test)
8. [How It Works](#how-it-works)
9. [API Endpoints](#api-endpoints)
10. [Running Tests](#running-tests)
11. [Adding or Editing FAQs](#adding-or-editing-faqs)
12. [Troubleshooting](#troubleshooting)

---

## Project Overview

**ShopEase FAQ Chatbot** answers frequently-asked questions about an online shopping platform. A user types a question in plain English; the chatbot finds the most semantically similar FAQ and returns the answer — entirely offline using local Python libraries.

**Key capabilities:**

- 35 curated FAQ entries across 8 categories (Orders, Returns, Payments, Account, Products, Shipping, Support, Offers)
- NLP preprocessing pipeline (lowercasing → punctuation removal → tokenization → stop-word removal → lemmatization)
- TF-IDF + cosine similarity matching with a configurable threshold
- Clean, responsive chat UI with sidebar navigation, suggested questions, and typing indicator
- RESTful Flask backend with proper error handling and HTTP status codes
- Automated test suite with 30+ test cases

---

## Technologies Used

| Layer        | Technology                             |
|--------------|----------------------------------------|
| Frontend     | HTML5, CSS3, Vanilla JavaScript (ES2022) |
| Backend      | Python 3.9+, Flask 3.x, Flask-CORS    |
| NLP          | NLTK (tokenizer, lemmatizer, POS tagger, stop words) |
| Matching     | scikit-learn (TfidfVectorizer, cosine_similarity) |
| Data         | JSON file (`faqs.json`)               |
| Testing      | pytest                                 |

---

## Project Structure

```
faq-chatbot/
├── app.py              # Flask backend (API routes)
├── faq_matcher.py      # TF-IDF + cosine similarity engine
├── preprocess.py       # NLP preprocessing pipeline
├── faqs.json           # FAQ dataset (35 Q&A pairs)
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── tests/
│   ├── conftest.py     # pytest path setup
│   └── test_chatbot.py # 30+ automated tests
└── frontend/
    ├── index.html      # Chat UI markup
    ├── style.css       # Styles (blue-purple theme, responsive)
    └── script.js       # Frontend logic (fetch, render, events)
```

---

## Setup and Installation

### Prerequisites

- **Python 3.9 or higher** — [Download Python](https://www.python.org/downloads/)
- **pip** (bundled with Python 3.9+)
- A modern web browser (Chrome, Firefox, Edge, Safari)

### 1 — Clone or Download the Project

```bash
# If using Git:
git clone [https://github.com/PrakashShrijal/CodeAlpha_faqChatbot]
cd faq-chatbot

# Or simply unzip the project and cd into it.
```

### 2 — Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

### 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: `flask`, `flask-cors`, `nltk`, `scikit-learn`, `numpy`, `pytest`.

### 4 — Download NLTK Data (Automatic)

NLTK resources are downloaded **automatically** the first time you start the server. If you want to pre-download them manually:

```python
python -c "import nltk; nltk.download(['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'omw-1.4'])"
```

---

## Running the Application

### Start the Flask Backend

```bash
# From the faq-chatbot/ root directory (with venv activated):
python app.py
```

You should see:

```
INFO – Loading FAQ Matcher...
INFO – Loaded 35 FAQs from faqs.json
INFO – TF-IDF index built: 35 questions × N features
INFO – Starting FAQ Chatbot server on http://127.0.0.1:5000
 * Running on http://0.0.0.0:5000
```

> **Note:** The first run may take 10–15 seconds while NLTK downloads its data packages. Subsequent runs are fast.

### Optional Environment Variables

| Variable       | Default | Description                               |
|----------------|---------|-------------------------------------------|
| `PORT`         | `5000`  | Port for the Flask server                 |
| `FLASK_DEBUG`  | `false` | Set to `true` for debug/auto-reload mode  |

```bash
# Example: custom port + debug mode
PORT=8080 FLASK_DEBUG=true python app.py
```

---

## Accessing the Chatbot

Open your browser and go to:

```
http://127.0.0.1:5000
```

The Flask server serves the frontend automatically — no separate web server needed.

---

## Example Questions to Test

### Exact matches (should score ≥ 80%)
```
How can I track my order?
What is the return policy?
What payment methods are accepted?
How do I contact customer support?
Do you offer free shipping?
```

### Paraphrased questions (should still match)
```
Where can I see my shipment status?
How long does it take to get my money back?
Is it possible to cancel a placed order?
I need help, who should I call?
How do I see my package status?
```

### Fallback triggers (should return "Sorry" message)
```
What is the capital of France?
Tell me a joke
How do I cook pasta?
```

---

## How It Works

### 1. NLP Text Preprocessing (`preprocess.py`)

Each FAQ question and every user query goes through the same pipeline:

| Step | Action | Example |
|------|--------|---------|
| Lowercase | Normalize case | `"Track My Order"` → `"track my order"` |
| Punctuation removal | Strip symbols | `"order!!!"` → `"order"` |
| Whitespace cleanup | Collapse spaces | `"  track  order  "` → `"track order"` |
| Tokenization | Split to words | `"track order"` → `["track", "order"]` |
| POS tagging | Tag each word | `"tracking"` → `VBG` (verb) |
| Lemmatization | Base form | `"tracking"` → `"track"` |
| Stop-word removal | Remove filler | `"the", "is", "at"` → removed |

**Why keep question words?** Words like "what", "how", "when" are kept because they convey the type of information the user seeks.

### 2. TF-IDF Vectorization

**TF-IDF** (Term Frequency × Inverse Document Frequency) converts text into numerical vectors:
- **TF** — how often a word appears in one document (question)
- **IDF** — how rare the word is across all FAQs (rarer = more informative)
- **bigrams** — word pairs like "track order" are also captured (ngram_range=(1,2))

Result: each FAQ question becomes a high-dimensional numeric vector.

### 3. Cosine Similarity

The user's question vector is compared against every FAQ vector using **cosine similarity** — the cosine of the angle between two vectors:

```
similarity = (A · B) / (|A| × |B|)   — ranges from 0 to 1
```

- `1.0` = identical direction (perfect match)
- `0.0` = completely unrelated

### 4. Similarity Threshold

The default threshold is **0.15**. If the best score is below this:

> "Sorry, I couldn't find a suitable answer. Could you rephrase your question?"

You can adjust the threshold in `faq_matcher.py` → `DEFAULT_THRESHOLD`.

---

## API Endpoints

### `GET /api/health`

Verifies the backend is running.

```json
{ "status": "ok", "message": "FAQ Chatbot API is running.", "faq_count": 35 }
```

### `GET /api/faqs`

Returns all FAQs. Supports `?category=Orders` filter.

```json
{
  "faqs": [{ "id": 1, "category": "Orders", "question": "...", "answer": "..." }],
  "total": 35,
  "categories": ["Account", "Offers", "Orders", ...]
}
```

### `POST /api/chat`

**Request:**
```json
{ "question": "How can I track my order?" }
```

**Response (matched):**
```json
{
  "answer": "You can track your order by logging into your account...",
  "matched_question": "How can I track my order?",
  "similarity_score": 0.82,
  "matched": true,
  "category": "Orders",
  "faq_id": 1
}
```

**Response (no match):**
```json
{
  "answer": "Sorry, I couldn't find a suitable answer...",
  "matched_question": "",
  "similarity_score": 0.04,
  "matched": false,
  "category": "",
  "faq_id": null
}
```

---

## Running Tests

```bash
# From the faq-chatbot/ root directory (with venv activated):
python -m pytest tests/ -v
```

Expected output: **30+ tests** across 6 categories — all should pass.

```
tests/test_chatbot.py::TestPreprocessing::test_lowercase          PASSED
tests/test_chatbot.py::TestExactMatch::test_track_order           PASSED
tests/test_chatbot.py::TestParaphraseMatch::test_paraphrase_refund PASSED
tests/test_chatbot.py::TestFallback::test_unrelated_triggers_fallback PASSED
...
30 passed in X.XXs
```

---

## Adding or Editing FAQs

Open `faqs.json` and add a new entry following this structure:

```json
{
  "id": 36,
  "category": "General",
  "question": "Do you have a mobile app?",
  "answer": "Yes! Download the ShopEase app on iOS and Android for a faster shopping experience."
}
```

**No code changes needed** — the matcher reloads from JSON on every server restart. To hot-reload without restarting, call `matcher.reload()` programmatically.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: flask` | Activate your venv: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux), then `pip install -r requirements.txt` |
| `LookupError: punkt` | Run `python -c "import nltk; nltk.download('punkt')"` |
| Port 5000 already in use | `PORT=5001 python app.py` |
| Frontend shows "Backend offline" | Ensure Flask is running: `python app.py` |
| All questions return fallback | Lower the threshold in `faq_matcher.py`: `DEFAULT_THRESHOLD = 0.10` |
| Slow first start | NLTK is downloading data packages (one-time, ~30 seconds) |
| `CORS` errors in browser | Flask-CORS is installed; clear browser cache and retry |
| Tests fail with import errors | Run pytest from the project root: `cd faq-chatbot && python -m pytest tests/ -v` |

---

## Project Summary

This project demonstrates:

- **Backend architecture**: modular Flask app with clean separation of concerns
- **NLP pipeline**: real preprocessing using NLTK lemmatization and POS tagging
- **Information retrieval**: TF-IDF vectorization with bigrams and cosine similarity
- **Frontend UX**: responsive chat interface with sidebar, suggestions, and typing feedback
- **Software quality**: automated tests covering preprocessing, matching, API, and edge cases
