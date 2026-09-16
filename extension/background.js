// Background service worker — relays messages between the content script
// (which scrapes chat turns off the page) and the local FastAPI server.

const SERVER_URL = "http://localhost:8000";

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "ADD_MESSAGE") {
    fetch(`${SERVER_URL}/add_message`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        conversation_id: message.conversationId,
        role: message.role,
        text: message.text,
      }),
    })
      .then((res) => res.json())
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true; // keep the message channel open for the async response
  }

  if (message.type === "RETRIEVE") {
    const params = new URLSearchParams({
      conversation_id: message.conversationId,
      query: message.query,
      top_k: message.topK || 5,
    });
    fetch(`${SERVER_URL}/retrieve?${params}`)
      .then((res) => res.json())
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true;
  }

  if (message.type === "EXPORT") {
    const params = new URLSearchParams({ conversation_id: message.conversationId });
    fetch(`${SERVER_URL}/export?${params}`)
      .then((res) => res.json())
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true;
  }
});
