// Content script — runs on each chatbot's page. Its job: detect new chat
// turns as they appear and send them to the background worker.
//
// IMPORTANT: each site's DOM structure is different and changes over time.
// The selectors below are PLACEHOLDERS — open devtools on each site
// (ChatGPT, Claude, Gemini), inspect the message containers, and replace
// these with real selectors. This file is the main "glue" work you'll need
// to do to get a working demo.

const SITE_SELECTORS = {
  "chat.openai.com": { messageContainer: "[data-message-author-role]", role: "data-message-author-role" },
  "chatgpt.com": { messageContainer: "[data-message-author-role]", role: "data-message-author-role" },
  // TODO: fill in real selectors after inspecting claude.ai's DOM
  "claude.ai": { messageContainer: ".TODO-claude-message-selector", role: null },
  // TODO: fill in real selectors after inspecting gemini.google.com's DOM
  "gemini.google.com": { messageContainer: ".TODO-gemini-message-selector", role: null },
};

function getConversationId() {
  // Simplest option: use the page URL as the conversation id. Good enough
  // to start with — refine if you need something more stable.
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
      if (response && response.ok) {
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
      seen.add(node);
      const role = config.role ? node.getAttribute(config.role) : "user";
      sendMessageToBackground(role, node.innerText);
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
