// Content script — runs on each chatbot's page. Its job: detect new chat
// turns as they appear and send them to the background worker.
//
// Only the USER's own messages are captured: those are where facts,
// preferences, goals and decisions come from. The assistant's replies are long,
// stream in token by token, and would mostly be stored as noise.
//
// IMPORTANT: each site's DOM structure is different and changes over time.
//   * chatgpt.com / chat.openai.com : selector below is the commonly used one,
//     verify it in devtools on your own account.
//   * claude.ai / gemini.google.com  : NOT filled in yet. Open devtools, inspect
//     a user message, and put a real selector in SITE_SELECTORS.

const CAPTURE_ROLES = ["user"];
const SETTLE_MS = 800; // wait a moment so a message's text is complete

const SITE_SELECTORS = {
  "chat.openai.com": { messageContainer: "[data-message-author-role]", role: "data-message-author-role" },
  "chatgpt.com": { messageContainer: "[data-message-author-role]", role: "data-message-author-role" },
  // TODO: fill in real selectors after inspecting claude.ai's DOM
  "claude.ai": { messageContainer: ".TODO-claude-message-selector", role: null },
  // TODO: fill in real selectors after inspecting gemini.google.com's DOM
  "gemini.google.com": { messageContainer: ".TODO-gemini-message-selector", role: null },
};

function getConversationId() {
  // Simplest option: use the page URL as the conversation id. A brand-new chat
  // changes its URL after the first message, so a fresh chat can end up split
  // across two ids. The popup's Conversation box can be used to line them up.
  return window.location.href;
}

function getSiteConfig() {
  const host = window.location.hostname;
  return SITE_SELECTORS[host] || null;
}

function sendMessageToBackground(role, text) {
  if (!text || text.trim().length === 0) return;
  chrome.runtime.sendMessage(
    {
      type: "ADD_MESSAGE",
      conversationId: getConversationId(),
      role: role || "user",
      text: text.trim(),
    },
    (response) => {
      if (chrome.runtime.lastError) {
        console.warn("[adaptive-memory] background worker not reachable:", chrome.runtime.lastError.message);
      } else if (response && response.ok) {
        console.debug("[adaptive-memory] processed:", response.data);
      } else {
        console.warn("[adaptive-memory] failed to process message:", response && response.error);
      }
    }
  );
}

function observeNewMessages() {
  const config = getSiteConfig();
  if (!config) {
    console.warn("[adaptive-memory] no selector config for this site yet — see content.js TODOs");
    return;
  }

  const seen = new WeakSet();

  const scanForNewMessages = () => {
    const nodes = document.querySelectorAll(config.messageContainer);
    nodes.forEach((node) => {
      if (seen.has(node)) return;
      const role = config.role ? node.getAttribute(config.role) : "user";
      if (!CAPTURE_ROLES.includes(role)) {
        seen.add(node); // assistant message: ignore it for good
        return;
      }
      seen.add(node);
      // Read the text after a short delay, once it has settled.
      setTimeout(() => sendMessageToBackground(role, node.innerText), SETTLE_MS);
    });
  };

  // Chat UIs render messages dynamically — a MutationObserver catches new
  // turns as they're added to the DOM.
  const observer = new MutationObserver(() => scanForNewMessages());
  observer.observe(document.body, { childList: true, subtree: true });

  // Also scan once on load in case messages are already present.
  scanForNewMessages();
}

observeNewMessages();