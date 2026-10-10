// Popup panel — a small memory console for one conversation:
//   1. add a message and see what the engine did with it
//   2. see everything it remembers
//   3. ask what is relevant to a question
//   + copy the memory as a prompt / JSON, or clear it.
//
// All text coming from chats is inserted with textContent, never innerHTML,
// because chat content is untrusted.

const CHAT_HOSTS = ["chat.openai.com", "chatgpt.com", "claude.ai", "gemini.google.com"];
const CATEGORY_LABELS = {
  fact: "Facts about me",
  preference: "My preferences",
  goal: "My goals",
  decision: "Decisions I made",
  completed_task: "Things already done",
};

const $ = (id) => document.getElementById(id);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function send(message) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage(message, (response) => {
      if (chrome.runtime.lastError) {
        resolve({ ok: false, error: chrome.runtime.lastError.message });
        return;
      }
      resolve(response || { ok: false, error: "no response" });
    });
  });
}

function conversationId() {
  return $("conv-id").value.trim() || "demo";
}

let statusTimer = null;
function setStatus(text, clearAfterMs = 4000) {
  $("status").textContent = text;
  clearTimeout(statusTimer);
  if (text && clearAfterMs) {
    statusTimer = setTimeout(() => ($("status").textContent = ""), clearAfterMs);
  }
}

// ---- server status --------------------------------------------------------
async function checkServer() {
  const res = await send({ type: "HEALTH" });
  const up = res.ok && res.data && res.data.status === "ok";
  $("server-dot").className = "dot " + (up ? "up" : "down");
  $("server-text").textContent = up ? "server connected" : "server not running";
  return up;
}

// ---- rendering --------------------------------------------------------------
function renderMemory(items, highlightText) {
  const list = $("memory-list");
  list.replaceChildren();
  $("count").textContent = items.length ? `(${items.length})` : "";

  if (items.length === 0) {
    list.appendChild(el("div", "empty", "Nothing stored yet for this conversation."));
    return;
  }
  items.forEach((item) => {
    const row = el("div", "memory-item" + (item.text === highlightText ? " flash" : ""));
    row.appendChild(el("span", "meta", Number(item.score).toFixed(2)));
    row.appendChild(el("span", "category", item.category));
    row.appendChild(document.createElement("br"));
    row.appendChild(document.createTextNode(item.text));
    list.appendChild(row);
  });
}

function renderResult(res) {
  const box = $("result");
  box.replaceChildren();

  if (!res.ok) {
    box.appendChild(el("span", "chip error", "error"));
    box.appendChild(el("span", "why", `${res.error} (is the server running on port 8000?)`));
    return;
  }
  const d = res.data;
  const action = d.action;
  box.appendChild(el("span", "chip " + action, action));

  let summary = "";
  if (action === "added") summary = `Saved as ${d.category}. `;
  if (action === "replaced") summary = `Updated an older memory (${d.category}). `;
  if (summary) box.appendChild(el("span", "", summary));

  const why = d.reasoning || d.reason;
  if (why) box.appendChild(el("span", "why", why));
  if (d.llm_error) {
    box.appendChild(document.createElement("br"));
    box.appendChild(el("span", "warn", "Warning: the contradiction check failed, so this was stored without it."));
  }
}

function renderRetrieved(results) {
  const box = $("retrieved");
  box.replaceChildren();
  if (results.length === 0) {
    box.appendChild(el("div", "empty", "Nothing relevant to that question."));
    return;
  }
  results.forEach((r) => {
    const row = el("div", "memory-item");
    row.appendChild(el("span", "meta", `sim ${Number(r.similarity).toFixed(2)}`));
    row.appendChild(el("span", "category", r.category));
    row.appendChild(document.createElement("br"));
    row.appendChild(document.createTextNode(r.text));
    box.appendChild(row);
  });
}

// ---- data -------------------------------------------------------------------
async function loadMemory() {
  const res = await send({ type: "EXPORT", conversationId: conversationId() });
  return res.ok ? res.data.memory || [] : null;
}

async function refreshMemory(highlightText) {
  const items = await loadMemory();
  if (items === null) {
    $("memory-list").replaceChildren(el("div", "empty", "Couldn't reach the server."));
    return null;
  }
  renderMemory(items, highlightText);
  return items;
}

function formatPrompt(items) {
  if (!items.length) return "";
  const lines = [
    "Here is what I have told you in earlier conversations. " +
      "Use it as background, and do not repeat it back unless it is relevant:",
  ];
  Object.keys(CATEGORY_LABELS).forEach((category) => {
    const group = items.filter((i) => i.category === category);
    if (!group.length) return;
    lines.push("", `${CATEGORY_LABELS[category]}:`);
    group.forEach((i) => lines.push(`- ${i.text}`));
  });
  return lines.join("\n");
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (e) {
    const area = document.createElement("textarea");
    area.value = text;
    document.body.appendChild(area);
    area.select();
    const ok = document.execCommand("copy");
    area.remove();
    return ok;
  }
}

// ---- actions ----------------------------------------------------------------
async function addMessage() {
  const text = $("msg").value.trim();
  if (!text) return;
  const button = $("add-btn");
  button.disabled = true;
  button.textContent = "Adding…";
  $("result").textContent = "Working: categorizing and checking for contradictions…";

  const res = await send({
    type: "ADD_MESSAGE",
    conversationId: conversationId(),
    role: "user",
    text,
  });
  renderResult(res);
  const stored = res.ok && (res.data.action === "added" || res.data.action === "replaced");
  if (stored) $("msg").value = "";
  await refreshMemory(stored ? text : undefined);

  button.disabled = false;
  button.textContent = "Add to memory";
}

async function askQuestion() {
  const query = $("query").value.trim();
  if (!query) return;
  const res = await send({ type: "RETRIEVE", conversationId: conversationId(), query, topK: 5 });
  if (!res.ok) {
    $("retrieved").replaceChildren(el("div", "empty", `${res.error} (is the server running?)`));
    return;
  }
  renderRetrieved(res.data.results || []);
}

async function copyAsPrompt() {
  const items = await loadMemory();
  if (!items || items.length === 0) {
    setStatus("Nothing to copy yet.");
    return;
  }
  const ok = await copyText(formatPrompt(items));
  setStatus(ok ? "Copied. Paste it at the start of a chat in any chatbot." : "Couldn't copy.");
}

async function copyJson() {
  const res = await send({ type: "EXPORT", conversationId: conversationId() });
  if (!res.ok) {
    setStatus("Couldn't reach the server.");
    return;
  }
  const ok = await copyText(JSON.stringify(res.data, null, 2));
  setStatus(ok ? "Snapshot JSON copied." : "Couldn't copy.");
}

let clearArmed = false;
async function clearConversation() {
  const button = $("clear-btn");
  if (!clearArmed) {
    clearArmed = true;
    button.textContent = "Click again to clear";
    setTimeout(() => {
      clearArmed = false;
      button.textContent = "Clear";
    }, 3000);
    return;
  }
  clearArmed = false;
  button.textContent = "Clear";
  const res = await send({ type: "CLEAR", conversationId: conversationId() });
  setStatus(res.ok ? "Memory cleared for this conversation." : "Couldn't clear.");
  $("result").replaceChildren();
  $("retrieved").replaceChildren();
  await refreshMemory();
}

// ---- startup ----------------------------------------------------------------
async function defaultConversationId() {
  try {
    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    const url = tabs[0] && tabs[0].url;
    if (url && CHAT_HOSTS.includes(new URL(url).hostname)) return url;
  } catch (e) {
    /* fall through to the saved / demo id */
  }
  try {
    return localStorage.getItem("conv-id") || "demo";
  } catch (e) {
    return "demo";
  }
}

async function init() {
  $("conv-id").value = await defaultConversationId();
  $("conv-id").addEventListener("change", () => {
    try {
      localStorage.setItem("conv-id", conversationId());
    } catch (e) {
      /* storage unavailable: fine */
    }
    $("result").replaceChildren();
    $("retrieved").replaceChildren();
    refreshMemory();
  });

  $("add-btn").addEventListener("click", addMessage);
  $("msg").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) addMessage();
  });
  $("ask-btn").addEventListener("click", askQuestion);
  $("query").addEventListener("keydown", (e) => {
    if (e.key === "Enter") askQuestion();
  });
  $("copy-prompt-btn").addEventListener("click", copyAsPrompt);
  $("export-btn").addEventListener("click", copyJson);
  $("clear-btn").addEventListener("click", clearConversation);

  const up = await checkServer();
  if (up) await refreshMemory();
  else $("memory-list").replaceChildren(el("div", "empty", "Start the server: uvicorn server:app --port 8000"));
}

init();