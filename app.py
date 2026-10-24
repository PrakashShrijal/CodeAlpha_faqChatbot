"""
app.py
------
Flask backend for the FAQ Chatbot.

Endpoints
---------
GET  /api/health  – Health check: confirms the server is running.
GET  /api/faqs    – Returns the full FAQ dataset as JSON.
POST /api/chat    – Accepts a user question and returns the best match.

The backend serves the frontend static files when accessed via the browser.
CORS is enabled so the frontend (even on a different port) can call the API.
"""

import logging
import os
import sys

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Make sure imports resolve whether app.py is run directly or from a subdir.
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from faq_matcher import FAQMatcher

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask app setup
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, "frontend"),
    static_url_path="",
)
CORS(app)   # Allow cross-origin requests from the frontend

# ---------------------------------------------------------------------------
# Initialise the FAQ matcher (loads JSON + builds TF-IDF index at startup)
# ---------------------------------------------------------------------------
FAQ_PATH = os.path.join(BASE_DIR, "faqs.json")

try:
    matcher = FAQMatcher(faq_path=FAQ_PATH, threshold=0.15)
    logger.info("FAQMatcher initialised successfully.")
except Exception as exc:
    logger.critical("Failed to initialise FAQMatcher: %s", exc)
    sys.exit(1)


# ===========================================================================
# Routes – Static frontend
# ===========================================================================

@app.route("/")
def serve_index():
    """Serve the chat UI (frontend/index.html)."""
    return send_from_directory(app.static_folder, "index.html")


# ===========================================================================
# Routes – API
# ===========================================================================

@app.route("/api/health", methods=["GET"])
def health_check():
    """
    GET /api/health
    Returns a simple JSON confirming the server is operational and shows
    how many FAQs are loaded.
    """
    return jsonify({
        "status": "ok",
        "message": "FAQ Chatbot API is running.",
        "faq_count": len(matcher.get_all_faqs()),
    }), 200


@app.route("/api/faqs", methods=["GET"])
def get_faqs():
    """
    GET /api/faqs
    Returns all FAQ entries.  Supports an optional ?category= query param
    to filter by category.

    Response
    --------
    {
        "faqs": [ { "id": 1, "category": "...", "question": "...", "answer": "..." }, ... ],
        "total": 35,
        "categories": ["Account", "Orders", ...]
    }
    """
    category_filter = request.args.get("category", "").strip().lower()
    faqs = matcher.get_all_faqs()

    if category_filter:
        faqs = [
            f for f in faqs
            if f.get("category", "").lower() == category_filter
        ]

    return jsonify({
        "faqs":       faqs,
        "total":      len(faqs),
        "categories": matcher.get_categories(),
    }), 200


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    POST /api/chat
    Accepts a user question and returns the closest FAQ match.

    Request body (JSON)
    -------------------
    { "question": "How can I track my order?" }

    Response (matched)
    ------------------
    {
        "answer":           "You can track your order from...",
        "matched_question": "How can I track my order?",
        "similarity_score": 0.82,
        "matched":          true,
        "category":         "Orders",
        "faq_id":           1
    }

    Response (no match)
    -------------------
    {
        "answer":           "Sorry, I couldn't find a suitable answer...",
        "matched_question": "",
        "similarity_score": 0.04,
        "matched":          false,
        "category":         "",
        "faq_id":           null
    }
    """
    # ---- Validate Content-Type ----
    if not request.is_json:
        return jsonify({"error": "Request must be JSON (Content-Type: application/json)."}), 415

    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"error": "Invalid JSON body."}), 400

    question = body.get("question", "")

    # ---- Validate question field ----
    if not isinstance(question, str):
        return jsonify({"error": "'question' must be a string."}), 422

    question = question.strip()
    if not question:
        return jsonify({
            "answer":           "Please enter a question so I can help you!",
            "matched_question": "",
            "similarity_score": 0.0,
            "matched":          False,
            "category":         "",
            "faq_id":           None,
        }), 200

    # ---- Enforce reasonable length limit ----
    if len(question) > 500:
        return jsonify({"error": "Question must be 500 characters or fewer."}), 422

    # ---- Run the matcher ----
    try:
        result = matcher.match(question)
    except Exception as exc:
        logger.exception("Matcher error for question '%s': %s", question, exc)
        return jsonify({"error": "Internal server error during matching."}), 500

    logger.info(
        "Q: '%s' | Matched=%s | Score=%.4f",
        question,
        result["matched"],
        result["similarity_score"],
    )
    return jsonify(result), 200


# ===========================================================================
# Error handlers
# ===========================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error."}), 500


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info("Starting FAQ Chatbot server on http://127.0.0.1:%d", port)
    app.run(host="0.0.0.0", port=port, debug=debug)
