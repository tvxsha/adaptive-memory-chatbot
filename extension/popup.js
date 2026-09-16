// Popup panel — shows the live memory state for the active tab's conversation.

function getActiveTabUrl(callback) {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    callback(tabs[0] ? tabs[0].url : null);
  });
}

function renderMemory(items) {
  const list = document.getElementById("memory-list");
  const status = document.getElementById("status");
  list.innerHTML = "";

  if (!items || items.length === 0) {
    status.textContent = "No memory stored yet for this conversation.";
    return;
  }

  status.textContent = `${items.length} memory item(s)`;
  items.forEach((item) => {
    const div = document.createElement("div");
    div.className = "memory-item";
    div.innerHTML = `<span class="category">${item.category}</span>` +
      `<span class="score">${item.score.toFixed(2)}</span><br>${item.text}`;
    list.appendChild(div);
  });
}

getActiveTabUrl((url) => {
  if (!url) return;
  // Uses a broad query against the conversation's own history as a stand-in
  // for "show me everything" — refine once retrieval is tuned.
  chrome.runtime.sendMessage(
    { type: "RETRIEVE", conversationId: url, query: "", topK: 20 },
    (response) => {
      if (response && response.ok) {
        renderMemory(response.data.results);
      } else {
        document.getElementById("status").textContent =
          "Couldn't reach the local server — is it running on port 8000?";
      }
    }
  );

  document.getElementById("export-btn").addEventListener("click", () => {
    chrome.runtime.sendMessage(
      { type: "EXPORT", conversationId: url },
      (response) => {
        if (response && response.ok) {
          navigator.clipboard.writeText(JSON.stringify(response.data, null, 2));
          document.getElementById("status").textContent = "Snapshot copied to clipboard!";
        }
      }
    );
  });
});
