// Background service worker — relays messages between the content script /
// popup and the local FastAPI server.

const SERVER_URL = "http://localhost:8000";

// Messages are sent to the server ONE AT A TIME. Opening an existing chat
// would otherwise fire dozens of requests at once, and every one of them makes
// several LLM calls, which trips the Groq rate limit.
let addQueue = Promise.resolve();
const GAP_BETWEEN_ADDS_MS = 1500;

function callServer(path, options) {
  return fetch(`${SERVER_URL}${path}`, options).then(async (res) => {
    if (!res.ok) {
      throw new Error(`server returned ${res.status}`);
    }
    return res.json();
  });
}

function reply(promise, sendResponse) {
  promise
    .then((data) => sendResponse({ ok: true, data }))
    .catch((err) => sendResponse({ ok: false, error: String(err) }));
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "ADD_MESSAGE") {
    const job = () =>
      callServer("/add_message", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          conversation_id: message.conversationId,
          role: message.role,
          text: message.text,
        }),
      });
    const result = addQueue.then(job);
    // Keep the queue moving even if one message fails, and pause briefly.
    addQueue = result
      .catch(() => {})
      .then(() => new Promise((r) => setTimeout(r, GAP_BETWEEN_ADDS_MS)));
    reply(result, sendResponse);
    return true; // keep the message channel open for the async response
  }

  if (message.type === "RETRIEVE") {
    const params = new URLSearchParams({
      conversation_id: message.conversationId,
      query: message.query,
      top_k: message.topK || 5,
    });
    reply(callServer(`/retrieve?${params}`), sendResponse);
    return true;
  }

  if (message.type === "EXPORT") {
    const params = new URLSearchParams({ conversation_id: message.conversationId });
    reply(callServer(`/export?${params}`), sendResponse);
    return true;
  }

  if (message.type === "CLEAR") {
    reply(
      callServer(`/conversation/${encodeURIComponent(message.conversationId)}`, {
        method: "DELETE",
      }),
      sendResponse
    );
    return true;
  }

  if (message.type === "HEALTH") {
    reply(callServer("/health"), sendResponse);
    return true;
  }
});