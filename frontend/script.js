/**
 * script.js  –  FAQ Chatbot Frontend
 *
 * Responsibilities:
 *   - Render welcome message and suggested questions.
 *   - Handle text input (keyboard / button).
 *   - POST user questions to the Flask /api/chat endpoint.
 *   - Display bot replies with similarity metadata.
 *   - Sidebar navigation & FAQ modal browser.
 *   - Typing indicator, auto-scroll, character counter.
 *   - Toast notifications for errors.
 *   - Clear conversation.
 */

'use strict';

// ── Configuration ──────────────────────────────────────────────────────────
const API_BASE    = '';          // empty = same origin (Flask serves both)
const CHAT_URL    = `${API_BASE}/api/chat`;
const FAQS_URL    = `${API_BASE}/api/faqs`;
const HEALTH_URL  = `${API_BASE}/api/health`;

// ── DOM References ─────────────────────────────────────────────────────────
const messagesContainer = document.getElementById('messagesContainer');
const questionInput     = document.getElementById('questionInput');
const sendBtn           = document.getElementById('sendBtn');
const clearBtn          = document.getElementById('clearBtn');
const charCount         = document.getElementById('charCount');
const faqCountBadge     = document.getElementById('faqCount');
const suggestionsScroll = document.getElementById('suggestionsScroll');
const menuBtn           = document.getElementById('menuBtn');
const sidebar           = document.getElementById('sidebar');
const sidebarClose      = document.getElementById('sidebarClose');
const navBtns           = document.querySelectorAll('.nav-btn');
const modalOverlay      = document.getElementById('modalOverlay');
const modalBody         = document.getElementById('modalBody');
const modalClose        = document.getElementById('modalClose');
const modalTitle        = document.getElementById('modalTitle');

// ── State ──────────────────────────────────────────────────────────────────
let isSending = false;     // prevent double-sends
let allFaqs   = [];        // cached FAQ list from /api/faqs

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  renderWelcome();
  loadFAQs();
  checkHealth();
  bindEvents();
});

// ── Health check ───────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const resp = await fetch(HEALTH_URL);
    if (!resp.ok) throw new Error('API not healthy');
  } catch {
    showToast('⚠️ Backend offline. Start Flask with: python app.py', 'error', 8000);
  }
}

// ── Load FAQs from API ─────────────────────────────────────────────────────
async function loadFAQs() {
  try {
    const resp = await fetch(FAQS_URL);
    if (!resp.ok) throw new Error('Failed to load FAQs');
    const data = await resp.json();
    allFaqs = data.faqs || [];
    faqCountBadge.textContent = `${data.total} FAQs loaded`;
  } catch {
    faqCountBadge.textContent = 'FAQs unavailable';
  }
}

// ── Welcome message ────────────────────────────────────────────────────────
function renderWelcome() {
  const topics = ['Orders', 'Returns', 'Payments', 'Shipping', 'Account', 'Offers'];
  const card = document.createElement('div');
  card.className = 'welcome-card';
  card.innerHTML = `
    <div class="welcome-icon">🤖</div>
    <h2 class="welcome-title">Welcome to ShopEase FAQ Chatbot!</h2>
    <p class="welcome-desc">
      Hi there! I can answer your questions about orders, payments, returns, shipping,
      and more. Type your question below or click a suggested topic to get started.
    </p>
    <div class="welcome-tags">
      ${topics.map(t => `<span class="welcome-tag">${t}</span>`).join('')}
    </div>
  `;
  messagesContainer.appendChild(card);
}

// ── Bind UI events ─────────────────────────────────────────────────────────
function bindEvents() {
  // Send button
  sendBtn.addEventListener('click', handleSend);

  // Enter key (Shift+Enter = newline)
  questionInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  // Character counter + auto-resize textarea
  questionInput.addEventListener('input', () => {
    const len = questionInput.value.length;
    charCount.textContent = `${len} / 500`;
    charCount.className = 'char-count' + (len > 450 ? ' warn' : '') + (len >= 500 ? ' error' : '');
    autoResizeTextarea();
  });

  // Clear conversation
  clearBtn.addEventListener('click', clearConversation);

  // Suggested question chips
  suggestionsScroll.addEventListener('click', e => {
    const chip = e.target.closest('.suggestion-chip');
    if (chip) {
      questionInput.value = chip.dataset.q;
      questionInput.dispatchEvent(new Event('input'));
      handleSend();
    }
  });

  // Sidebar toggle (mobile)
  menuBtn.addEventListener('click', () => sidebar.classList.add('open'));
  sidebarClose.addEventListener('click', () => sidebar.classList.remove('open'));

  // Category nav buttons → open modal
  navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      navBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      sidebar.classList.remove('open');
      const category = btn.dataset.category;
      openFAQModal(category === 'all' ? null : category);
    });
  });

  // Close modal
  modalClose.addEventListener('click', closeModal);
  modalOverlay.addEventListener('click', e => {
    if (e.target === modalOverlay) closeModal();
  });

  // Keyboard: Escape closes modal / sidebar
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
      closeModal();
      sidebar.classList.remove('open');
    }
  });
}

// ── Auto-resize textarea ───────────────────────────────────────────────────
function autoResizeTextarea() {
  questionInput.style.height = 'auto';
  questionInput.style.height = Math.min(questionInput.scrollHeight, 120) + 'px';
}

// ── Send message flow ──────────────────────────────────────────────────────
async function handleSend() {
  const question = questionInput.value.trim();

  if (!question) {
    shakeInput();
    return;
  }

  if (isSending) return;
  isSending = true;

  // Render user bubble
  appendUserMessage(question);

  // Clear input
  questionInput.value = '';
  questionInput.style.height = 'auto';
  charCount.textContent = '0 / 500';
  charCount.className = 'char-count';

  // Show typing indicator
  const typingEl = showTypingIndicator();
  scrollToBottom();

  try {
    const resp = await fetch(CHAT_URL, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ question }),
    });

    // Small artificial delay (300–600 ms) so the typing indicator feels natural
    await delay(300 + Math.random() * 300);

    typingEl.remove();

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      appendBotMessage(
        `⚠️ Server error (${resp.status}): ${err.error || 'Something went wrong.'}`,
        { matched: false, similarity_score: 0 }
      );
      return;
    }

    const data = await resp.json();
    appendBotMessage(data.answer, data);

  } catch (fetchErr) {
    typingEl.remove();
    appendBotMessage(
      '⚠️ Could not reach the server. Please make sure the Flask backend is running.',
      { matched: false, similarity_score: 0 }
    );
    showToast('Connection error. Is Flask running?', 'error');
  } finally {
    isSending = false;
    scrollToBottom();
  }
}

// ── Render helpers ─────────────────────────────────────────────────────────

function appendUserMessage(text) {
  const row = document.createElement('div');
  row.className = 'message-row user';
  row.innerHTML = `
    <div class="msg-avatar user-av" aria-hidden="true">👤</div>
    <div class="bubble-wrap">
      <div class="bubble user">${escapeHTML(text)}</div>
      <span class="bubble-time">${formatTime()}</span>
    </div>
  `;
  messagesContainer.appendChild(row);
  scrollToBottom();
}

function appendBotMessage(answer, meta = {}) {
  const {
    matched          = false,
    similarity_score = 0,
    matched_question = '',
    category         = '',
  } = meta;

  const scorePercent  = Math.round(similarity_score * 100);
  const matchedClass  = matched ? 'matched'   : 'unmatched';
  const matchedLabel  = matched ? '✓ Matched' : '✗ No match';

  const metaHTML = `
    <div class="bubble-meta">
      <span class="match-badge ${matchedClass}">${matchedLabel}</span>
      ${matched ? `<span class="score-pill">Score: ${scorePercent}%</span>` : ''}
      ${category ? `<span class="category-tag">${escapeHTML(category)}</span>` : ''}
    </div>
    ${matched && matched_question ? `
      <div class="bubble-meta" title="Best-matched FAQ question" style="font-size:.7rem;color:var(--text-muted);padding:0 4px;margin-top:-2px">
        ↳ "${escapeHTML(matched_question)}"
      </div>` : ''}
  `;

  const row = document.createElement('div');
  row.className = 'message-row bot';
  row.innerHTML = `
    <div class="msg-avatar bot-av" aria-hidden="true">🤖</div>
    <div class="bubble-wrap">
      <div class="bubble bot">${escapeHTML(answer)}</div>
      ${metaHTML}
      <span class="bubble-time">${formatTime()}</span>
    </div>
  `;
  messagesContainer.appendChild(row);
}

function showTypingIndicator() {
  const row = document.createElement('div');
  row.className = 'message-row bot typing-row';
  row.id = 'typingIndicator';
  row.innerHTML = `
    <div class="msg-avatar bot-av" aria-hidden="true">🤖</div>
    <div class="typing-bubble">
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
    </div>
  `;
  messagesContainer.appendChild(row);
  return row;
}

// ── Clear conversation ─────────────────────────────────────────────────────
function clearConversation() {
  messagesContainer.innerHTML = '';
  renderWelcome();
  showToast('Conversation cleared.', 'success');
}

// ── FAQ Modal ──────────────────────────────────────────────────────────────
function openFAQModal(category) {
  const filtered = category
    ? allFaqs.filter(f => f.category === category)
    : allFaqs;

  modalTitle.textContent = category ? `${category} FAQs` : 'All FAQs';
  modalBody.innerHTML = '';

  if (!filtered.length) {
    modalBody.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:24px">No FAQs found.</p>';
  } else {
    filtered.forEach(faq => {
      const item = document.createElement('div');
      item.className = 'faq-item';
      item.innerHTML = `
        <div class="faq-question">${escapeHTML(faq.question)}</div>
        <div class="faq-category-label">${escapeHTML(faq.category || '')}</div>
      `;
      item.addEventListener('click', () => {
        closeModal();
        questionInput.value = faq.question;
        questionInput.dispatchEvent(new Event('input'));
        handleSend();
      });
      modalBody.appendChild(item);
    });
  }

  modalOverlay.classList.add('open');
}

function closeModal() {
  modalOverlay.classList.remove('open');
}

// ── Toast notifications ────────────────────────────────────────────────────
function showToast(message, type = 'info', duration = 3500) {
  // Remove existing toast
  document.querySelectorAll('.toast').forEach(t => t.remove());

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  document.body.appendChild(toast);

  requestAnimationFrame(() => {
    requestAnimationFrame(() => toast.classList.add('show'));
  });

  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 350);
  }, duration);
}

// ── Input shake animation (empty submit) ───────────────────────────────────
function shakeInput() {
  const wrapper = questionInput.closest('.input-wrapper');
  wrapper.style.animation = 'none';
  wrapper.offsetHeight;   // reflow
  wrapper.style.animation = 'shake .4s ease';
}

// Inject shake keyframes once
(function injectShakeCSS() {
  const style = document.createElement('style');
  style.textContent = `
    @keyframes shake {
      0%,100% { transform: translateX(0); }
      20%      { transform: translateX(-6px); }
      40%      { transform: translateX( 6px); }
      60%      { transform: translateX(-4px); }
      80%      { transform: translateX( 4px); }
    }
  `;
  document.head.appendChild(style);
})();

// ── Utilities ──────────────────────────────────────────────────────────────

/** Scroll the message list to the very bottom. */
function scrollToBottom() {
  messagesContainer.scrollTo({
    top:      messagesContainer.scrollHeight,
    behavior: 'smooth',
  });
}

/** Escape HTML to prevent XSS when rendering user content. */
function escapeHTML(str) {
  const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
  return String(str).replace(/[&<>"']/g, m => map[m]);
}

/** Format current time as HH:MM. */
function formatTime() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/** Promisified setTimeout. */
function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
