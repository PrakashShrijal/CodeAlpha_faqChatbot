"""
preprocess.py
-------------
NLP Text Preprocessing Pipeline for the FAQ Chatbot.

This module handles all text normalization steps before vectorization:
  1. Lowercasing        – ensures "Order" and "order" match identically.
  2. Punctuation removal – strips symbols that add no semantic value.
  3. Whitespace cleanup – collapses multiple spaces / tabs / newlines.
  4. Tokenization       – splits a sentence into individual word tokens.
  5. Stop-word removal  – drops common filler words (e.g. "the", "is").
  6. Lemmatization      – reduces words to their dictionary base form
                          (e.g. "tracking" → "track", "orders" → "order").

Why lemmatization over stemming?
  Lemmatization produces real English words using vocabulary & morphological
  analysis, which preserves meaning better inside TF-IDF vectors.
"""

import re
import string
import logging

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK resource bootstrap
# Download only what is missing so startup stays fast on repeated runs.
# ---------------------------------------------------------------------------
_NLTK_PACKAGES = [
    ("tokenizers/punkt",          "punkt"),
    ("tokenizers/punkt_tab",      "punkt_tab"),
    ("corpora/stopwords",         "stopwords"),
    ("corpora/wordnet",           "wordnet"),
    ("taggers/averaged_perceptron_tagger",       "averaged_perceptron_tagger"),
    ("taggers/averaged_perceptron_tagger_eng",   "averaged_perceptron_tagger_eng"),
    ("corpora/omw-1.4",           "omw-1.4"),
]

for _path, _pkg in _NLTK_PACKAGES:
    try:
        nltk.data.find(_path)
    except LookupError:
        logger.info("Downloading NLTK resource: %s", _pkg)
        nltk.download(_pkg, quiet=True)


# ---------------------------------------------------------------------------
# Module-level singletons (created once, reused on every call)
# ---------------------------------------------------------------------------
_lemmatizer  = WordNetLemmatizer()
_stop_words  = set(stopwords.words("english"))

# Keep question-words and negations – they affect meaning in FAQs.
_KEEP_WORDS = {
    "what", "when", "where", "which", "who", "whom", "whose",
    "why", "how", "no", "not", "nor", "never", "neither",
    "can", "cannot", "could", "should", "would", "will",
}
_STOP_WORDS_FILTERED = _stop_words - _KEEP_WORDS


def _get_wordnet_pos(treebank_tag: str) -> str:
    """
    Convert a Penn-Treebank POS tag to a WordNet POS constant.
    Lemmatization is more accurate when the POS is specified.
    Defaults to NOUN when the tag is unrecognised.
    """
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN   # default


def preprocess(text: str) -> str:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    text : str
        Raw user input or FAQ question.

    Returns
    -------
    str
        A cleaned, lemmatized, stop-word-filtered string ready for
        TF-IDF vectorization.

    Steps
    -----
    1. Guard against empty / non-string input.
    2. Lowercase.
    3. Remove punctuation (keep hyphens inside words).
    4. Collapse whitespace.
    5. Tokenize.
    6. POS-tag tokens for accurate lemmatization.
    7. Lemmatize each token.
    8. Remove stop-words (preserving question-words and negations).
    9. Rejoin into a single string.
    """

    # --- Step 1: Guard ---
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if not text:
        return ""

    # --- Step 2: Lowercase ---
    text = text.lower()

    # --- Step 3: Remove punctuation (keep apostrophe for contractions) ---
    # Replace punctuation with a space so "order,track" → "order track"
    text = re.sub(r"[^\w\s'-]", " ", text)
    # Remove standalone apostrophes / hyphens left after the above
    text = re.sub(r"['\-]{1,}", " ", text)

    # --- Step 4: Collapse whitespace ---
    text = re.sub(r"\s+", " ", text).strip()

    # --- Step 5: Tokenize ---
    tokens = word_tokenize(text)

    # --- Step 6: POS-tag for accurate lemmatization ---
    pos_tags = pos_tag(tokens)

    # --- Steps 7 & 8: Lemmatize + stop-word filter ---
    processed_tokens = []
    for token, tag in pos_tags:
        if not token.isalpha():          # skip pure-number / symbol tokens
            continue
        wn_pos = _get_wordnet_pos(tag)
        lemma  = _lemmatizer.lemmatize(token, wn_pos)
        if lemma not in _STOP_WORDS_FILTERED and len(lemma) > 1:
            processed_tokens.append(lemma)

    # --- Step 9: Rejoin ---
    return " ".join(processed_tokens)


def preprocess_batch(texts: list) -> list:
    """
    Convenience wrapper to preprocess a list of strings.
    Returns a list of cleaned strings in the same order.
    """
    return [preprocess(t) for t in texts]
