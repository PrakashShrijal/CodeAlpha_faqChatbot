"""
tests/test_chatbot.py
---------------------
Automated test suite for the FAQ Chatbot.

Run with:   python -m pytest tests/ -v
            (from the faq-chatbot/ root directory)

Test categories
---------------
1. Preprocessing unit tests   – verifies the NLP pipeline transforms text correctly.
2. Exact-match tests          – a question from the FAQ should score highly.
3. Paraphrase tests           – rephrased versions should still match.
4. Threshold / fallback tests – unrelated questions must trigger the fallback.
5. Edge-case tests            – empty input, symbols-only, very long input.
6. API integration tests      – Flask test client checks HTTP responses.
"""

import sys
import os
import json
import pytest

# ---------------------------------------------------------------------------
# Path setup so imports work regardless of how pytest is invoked
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from preprocess import preprocess, preprocess_batch
from faq_matcher import FAQMatcher, FALLBACK_MESSAGE
from app import app as flask_app


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def matcher():
    """A single FAQMatcher instance shared across the module."""
    faq_path = os.path.join(ROOT, "faqs.json")
    return FAQMatcher(faq_path=faq_path, threshold=0.15)


@pytest.fixture(scope="module")
def client():
    """Flask test client."""
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


# ===========================================================================
# 1. Preprocessing tests
# ===========================================================================

class TestPreprocessing:
    def test_lowercase(self):
        result = preprocess("HOW CAN I TRACK MY ORDER?")
        assert result == result.lower()

    def test_punctuation_removed(self):
        result = preprocess("Hello!!! How are you???")
        assert "!" not in result and "?" not in result

    def test_empty_string(self):
        assert preprocess("") == ""

    def test_whitespace_only(self):
        assert preprocess("   ") == ""

    def test_none_input(self):
        # preprocess should handle non-string gracefully
        assert preprocess(None) == ""  # type: ignore[arg-type]

    def test_numbers_filtered(self):
        result = preprocess("12345 track order 999")
        # pure numeric tokens should be dropped
        assert "12345" not in result
        assert "999" not in result

    def test_lemmatization(self):
        result = preprocess("tracking orders refunded")
        # "orders" → "order", "tracking" → "track", "refunded" → "refund"
        assert "order" in result
        assert "track" in result
        assert "refund" in result

    def test_batch_preprocessing(self):
        texts   = ["Track my order", "Return policy", ""]
        results = preprocess_batch(texts)
        assert len(results) == 3
        assert results[2] == ""   # empty stays empty

    def test_stopwords_removed(self):
        result = preprocess("how do I find the best product for me")
        # "the", "for", "do" should be removed
        assert "the" not in result.split()
        assert "for" not in result.split()


# ===========================================================================
# 2. Exact-match tests
# ===========================================================================

class TestExactMatch:
    def test_track_order(self, matcher):
        result = matcher.match("How can I track my order?")
        assert result["matched"] is True
        assert result["similarity_score"] >= 0.50

    def test_return_policy(self, matcher):
        result = matcher.match("What is the return policy?")
        assert result["matched"] is True
        assert "30" in result["answer"] or "return" in result["answer"].lower()

    def test_cancel_order(self, matcher):
        result = matcher.match("Can I cancel my order after placing it?")
        assert result["matched"] is True
        assert result["similarity_score"] >= 0.50

    def test_payment_methods(self, matcher):
        result = matcher.match("What payment methods are accepted?")
        assert result["matched"] is True

    def test_forgot_password(self, matcher):
        result = matcher.match("I forgot my password. How do I reset it?")
        assert result["matched"] is True

    def test_result_has_required_fields(self, matcher):
        result = matcher.match("How do I contact customer support?")
        for field in ("answer", "matched_question", "similarity_score", "matched", "category", "faq_id"):
            assert field in result, f"Missing field: {field}"


# ===========================================================================
# 3. Paraphrase tests
# ===========================================================================

class TestParaphraseMatch:
    def test_paraphrase_track_order(self, matcher):
        result = matcher.match("Where can I see my shipment status?")
        assert result["matched"] is True

    def test_paraphrase_refund(self, matcher):
        result = matcher.match("How long does it take to get my money back?")
        assert result["matched"] is True

    def test_paraphrase_cancel(self, matcher):
        result = matcher.match("Is it possible to cancel a placed order?")
        assert result["matched"] is True

    def test_paraphrase_delivery_time(self, matcher):
        result = matcher.match("How many days will my package take to arrive?")
        assert result["matched"] is True

    def test_paraphrase_contact_support(self, matcher):
        result = matcher.match("I need help. Who should I call?")
        # May or may not match, but must not crash
        assert isinstance(result["matched"], bool)

    def test_paraphrase_free_shipping(self, matcher):
        result = matcher.match("Do I have to pay for delivery?")
        assert result["matched"] is True


# ===========================================================================
# 4. Threshold / fallback tests
# ===========================================================================

class TestFallback:
    def test_unrelated_question_triggers_fallback(self, matcher):
        result = matcher.match("What is the capital of France?")
        assert result["matched"] is False
        assert result["answer"] == FALLBACK_MESSAGE

    def test_random_gibberish_triggers_fallback(self, matcher):
        result = matcher.match("xkcd banana pineapple moon cheese")
        assert result["matched"] is False

    def test_fallback_score_is_low(self, matcher):
        result = matcher.match("dinosaur asteroid quantum physics")
        assert result["similarity_score"] < 0.15

    def test_threshold_boundary(self):
        """A matcher with threshold=1.0 should NEVER match anything."""
        faq_path = os.path.join(ROOT, "faqs.json")
        strict_matcher = FAQMatcher(faq_path=faq_path, threshold=1.0)
        result = strict_matcher.match("How can I track my order?")
        assert result["matched"] is False


# ===========================================================================
# 5. Edge-case tests
# ===========================================================================

class TestEdgeCases:
    def test_empty_question(self, matcher):
        result = matcher.match("")
        assert result["matched"] is False
        assert result["similarity_score"] == 0.0

    def test_whitespace_only_question(self, matcher):
        result = matcher.match("     ")
        assert result["matched"] is False

    def test_symbols_only(self, matcher):
        result = matcher.match("??? !!! @@@")
        assert result["matched"] is False

    def test_very_long_question(self, matcher):
        long_q = "track order " * 50
        result = matcher.match(long_q)
        assert isinstance(result, dict)

    def test_single_word(self, matcher):
        result = matcher.match("refund")
        # single meaningful word may or may not match; must not raise
        assert isinstance(result["matched"], bool)

    def test_numeric_only(self, matcher):
        result = matcher.match("12345 67890")
        assert isinstance(result["matched"], bool)

    def test_non_english(self, matcher):
        result = matcher.match("Bonjour comment ça va?")
        assert isinstance(result["matched"], bool)


# ===========================================================================
# 6. API integration tests (Flask test client)
# ===========================================================================

class TestAPI:
    # ---- /api/health -------------------------------------------------------

    def test_health_status_code(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_health_response_structure(self, client):
        resp = client.get("/api/health")
        data = resp.get_json()
        assert data["status"] == "ok"
        assert "faq_count" in data

    # ---- /api/faqs ---------------------------------------------------------

    def test_faqs_returns_list(self, client):
        resp = client.get("/api/faqs")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data["faqs"], list)
        assert data["total"] >= 30

    def test_faqs_category_filter(self, client):
        resp = client.get("/api/faqs?category=orders")
        assert resp.status_code == 200
        data = resp.get_json()
        for faq in data["faqs"]:
            assert faq["category"].lower() == "orders"

    def test_faqs_contains_categories(self, client):
        resp = client.get("/api/faqs")
        data = resp.get_json()
        assert "categories" in data
        assert isinstance(data["categories"], list)

    # ---- /api/chat ---------------------------------------------------------

    def test_chat_valid_question(self, client):
        resp = client.post(
            "/api/chat",
            json={"question": "How can I track my order?"},
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["matched"] is True
        assert "answer" in data
        assert "similarity_score" in data
        assert "matched_question" in data

    def test_chat_empty_question(self, client):
        resp = client.post(
            "/api/chat",
            json={"question": ""},
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["matched"] is False

    def test_chat_missing_question_field(self, client):
        resp = client.post(
            "/api/chat",
            json={},
            content_type="application/json",
        )
        # Empty string treated as no-match, still 200
        assert resp.status_code == 200

    def test_chat_wrong_content_type(self, client):
        resp = client.post(
            "/api/chat",
            data="How can I track my order?",
            content_type="text/plain",
        )
        assert resp.status_code == 415

    def test_chat_question_too_long(self, client):
        resp = client.post(
            "/api/chat",
            json={"question": "track " * 200},
            content_type="application/json",
        )
        assert resp.status_code == 422

    def test_chat_fallback_response(self, client):
        resp = client.post(
            "/api/chat",
            json={"question": "What is the meaning of life in terms of pizza toppings?"},
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["matched"] is False
        assert FALLBACK_MESSAGE in data["answer"]

    def test_chat_response_score_range(self, client):
        resp = client.post(
            "/api/chat",
            json={"question": "What payment methods do you accept?"},
            content_type="application/json",
        )
        data = resp.get_json()
        assert 0.0 <= data["similarity_score"] <= 1.0

    def test_chat_similarity_score_type(self, client):
        resp = client.post(
            "/api/chat",
            json={"question": "return policy"},
            content_type="application/json",
        )
        data = resp.get_json()
        assert isinstance(data["similarity_score"], float)

    # ---- 404 / 405 ---------------------------------------------------------

    def test_unknown_route_returns_404(self, client):
        resp = client.get("/api/nonexistent")
        assert resp.status_code == 404

    def test_chat_get_not_allowed(self, client):
        resp = client.get("/api/chat")
        assert resp.status_code == 405
