"""
faq_matcher.py
--------------
Core FAQ matching engine.

Algorithm
---------
1. Load all FAQ questions from faqs.json.
2. Preprocess each question through the NLP pipeline (preprocess.py).
3. Fit a TF-IDF vectorizer on the preprocessed questions.
4. At query time:
   a. Preprocess the user's question.
   b. Transform it with the same fitted vectorizer.
   c. Compute cosine similarity against every stored question vector.
   d. Return the best-matching answer if similarity ≥ threshold,
      otherwise return the fallback response.

Why TF-IDF + cosine similarity?
--------------------------------
TF-IDF (Term Frequency – Inverse Document Frequency) weights words by how
unique they are across the corpus, so domain-specific terms like "refund" or
"tracking" get higher scores than common words.  Cosine similarity then
measures the directional closeness of two vectors, making it scale-invariant
(query length doesn't skew scores).
"""

import json
import logging
import os
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import preprocess, preprocess_batch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLD = 0.15   # minimum cosine similarity to consider a match
FALLBACK_MESSAGE  = (
    "Sorry, I couldn't find a suitable answer. "
    "Could you rephrase your question?"
)


class FAQMatcher:
    """
    Loads FAQ data, builds TF-IDF vectors, and answers user queries.

    Attributes
    ----------
    threshold : float
        Cosine similarity score below which no match is returned.
    faqs : list[dict]
        Raw FAQ records loaded from JSON.
    questions : list[str]
        Original (un-preprocessed) FAQ question strings.
    preprocessed_questions : list[str]
        Preprocessed FAQ question strings used for vectorization.
    vectorizer : TfidfVectorizer
        Fitted scikit-learn TF-IDF vectorizer.
    tfidf_matrix : scipy.sparse matrix
        TF-IDF matrix where each row is one FAQ question vector.
    """

    def __init__(
        self,
        faq_path: str = "faqs.json",
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.threshold = threshold
        self.faqs: list = []
        self.questions: list = []
        self.preprocessed_questions: list = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

        self._load_faqs(faq_path)
        self._build_index()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_faqs(self, path: str) -> None:
        """Read FAQs from a JSON file and validate structure."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"FAQ file not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        if not isinstance(data, list) or not data:
            raise ValueError("faqs.json must contain a non-empty JSON array.")

        for item in data:
            if "question" not in item or "answer" not in item:
                raise ValueError(
                    f"FAQ entry missing 'question' or 'answer': {item}"
                )

        self.faqs      = data
        self.questions = [faq["question"] for faq in data]
        logger.info("Loaded %d FAQs from %s", len(self.faqs), path)

    def _build_index(self) -> None:
        """
        Preprocess all questions and fit the TF-IDF vectorizer.

        TF-IDF parameters chosen for a small FAQ corpus:
        - ngram_range=(1, 2)  : captures both unigrams ("track") and
                                 bigrams ("track order") for richer context.
        - min_df=1            : include every term (corpus is small).
        - sublinear_tf=True   : apply log normalization to TF scores so
                                 frequent terms don't dominate unfairly.
        - analyzer='word'     : word-level tokenization (after preprocessing).
        """
        self.preprocessed_questions = preprocess_batch(self.questions)

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            analyzer="word",
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.preprocessed_questions
        )
        logger.info(
            "TF-IDF index built: %d questions × %d features",
            self.tfidf_matrix.shape[0],
            self.tfidf_matrix.shape[1],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(self, user_question: str) -> dict:
        """
        Find the most similar FAQ for a given user question.

        Parameters
        ----------
        user_question : str
            The raw question typed by the user.

        Returns
        -------
        dict with keys:
            answer          – the FAQ answer string (or fallback message).
            matched_question – the original FAQ question that was matched
                               (empty string on no-match).
            similarity_score – the cosine similarity score (0.0–1.0).
            matched          – True if a match above threshold was found.
            category         – the FAQ category (empty on no-match).
            faq_id           – the FAQ record id (None on no-match).
        """

        # Guard: empty input
        if not user_question or not user_question.strip():
            return {
                "answer": "Please enter a question so I can help you!",
                "matched_question": "",
                "similarity_score": 0.0,
                "matched": False,
                "category": "",
                "faq_id": None,
            }

        # Preprocess the query with the same pipeline used for FAQ questions
        processed_query = preprocess(user_question)

        # Edge case: preprocessing removed everything (e.g. all punctuation)
        if not processed_query.strip():
            return {
                "answer": FALLBACK_MESSAGE,
                "matched_question": "",
                "similarity_score": 0.0,
                "matched": False,
                "category": "",
                "faq_id": None,
            }

        # Vectorize the query using the FITTED vectorizer (no re-fitting)
        query_vector = self.vectorizer.transform([processed_query])

        # Compute cosine similarity between the query and ALL FAQ vectors
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Find the FAQ index with the highest similarity score
        best_idx   = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        logger.debug(
            "Query: '%s' | Best match idx=%d score=%.4f",
            user_question,
            best_idx,
            best_score,
        )

        # Apply the similarity threshold
        if best_score < self.threshold:
            return {
                "answer": FALLBACK_MESSAGE,
                "matched_question": "",
                "similarity_score": round(best_score, 4),
                "matched": False,
                "category": "",
                "faq_id": None,
            }

        best_faq = self.faqs[best_idx]
        return {
            "answer":           best_faq["answer"],
            "matched_question": best_faq["question"],
            "similarity_score": round(best_score, 4),
            "matched":          True,
            "category":         best_faq.get("category", ""),
            "faq_id":           best_faq.get("id"),
        }

    def get_all_faqs(self) -> list:
        """Return the raw FAQ list (used by the /api/faqs endpoint)."""
        return self.faqs

    def get_categories(self) -> list:
        """Return a sorted list of unique FAQ categories."""
        return sorted({faq.get("category", "General") for faq in self.faqs})

    def reload(self, faq_path: str = "faqs.json") -> None:
        """Hot-reload FAQs and rebuild the TF-IDF index without restarting."""
        self._load_faqs(faq_path)
        self._build_index()
        logger.info("FAQ index reloaded.")
