/* SmartDoc UI.
 *
 * Chat history lives in Postgres, not here. The only thing this file remembers
 * between renders is which thread is open — every message on screen is read
 * back from the API, so nothing can drift out of sync with the database.
 */

const API = "/api";

// null means "a new chat that hasn't been saved yet". The thread row is only
// created when the first question actually arrives, so empty chats never pile
// up in the rail.
let activeThreadId = null;

// Which documents to search in. Empty = search everything. This is a UI filter,
// not conversation state, so keeping it in the page is fine.
const scope = new Set();

// Last lists fetched from the API, kept only so a click can re-render the rail
// without a round trip.
let threads = [];
let documents = [];

let asking = false;

const el = {
  threads: document.getElementById("threads"),
  documents: document.getElementById("documents"),
  docHint: document.getElementById("doc-hint"),
  note: document.getElementById("note"),
  file: document.getElementById("file"),
  addPdf: document.getElementById("add-pdf"),
  newChat: document.getElementById("new-chat"),
  stream: document.getElementById("stream"),
  streamInner: document.getElementById("stream-inner"),
  composer: document.getElementById("composer"),
  question: document.getElementById("question"),
  send: document.getElementById("send"),
  scope: document.getElementById("scope"),
};

/* ── API ──────────────────────────────────────────────────────────────── */

async function request(path, options) {
  const response = await fetch(API + path, options);
  const body = await response.text();

  if (!response.ok) {
    // FastAPI puts the reason in {"detail": ...}; anything else comes through
    // as-is rather than being swallowed.
    let detail = body;
    try {
      detail = JSON.parse(body).detail ?? body;
    } catch {}
    throw new Error(detail);
  }

  return body ? JSON.parse(body) : null;
}

/* ── Small DOM helpers ────────────────────────────────────────────────── */

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function escapeHtml(text) {
  return text.replace(
    /[&<>"]/g,
    (character) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[character],
  );
}

/* The model answers in light markdown. Rendering the handful of marks it
 * actually uses keeps the page dependency-free; anything else stays as the
 * plain text it already is. Escaping happens first, so nothing in an answer
 * can inject markup. */
function renderMarkdown(text) {
  const inline = (line) =>
    escapeHtml(line)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>");

  const out = [];
  let paragraph = [];
  let list = null;

  const closeParagraph = () => {
    if (paragraph.length) {
      out.push(`<p>${paragraph.map(inline).join("<br>")}</p>`);
      paragraph = [];
    }
  };
  const closeList = () => {
    if (list) {
      out.push(`</${list}>`);
      list = null;
    }
  };

  for (const line of text.split("\n")) {
    const bullet = line.match(/^\s*[-*]\s+(.+)/);
    const numbered = line.match(/^\s*\d+\.\s+(.+)/);
    const heading = line.match(/^#{1,6}\s+(.+)/);

    if (bullet || numbered) {
      closeParagraph();
      const wanted = bullet ? "ul" : "ol";
      if (list !== wanted) {
        closeList();
        out.push(`<${wanted}>`);
        list = wanted;
      }
      out.push(`<li>${inline((bullet || numbered)[1])}</li>`);
    } else if (heading) {
      closeParagraph();
      closeList();
      out.push(`<h3>${inline(heading[1])}</h3>`);
    } else if (!line.trim()) {
      closeParagraph();
      closeList();
    } else {
      closeList();
      paragraph.push(line);
    }
  }

  closeParagraph();
  closeList();
  return out.join("");
}

/* ── Rail ─────────────────────────────────────────────────────────────── */

function drawThreads() {
  if (!threads.length) {
    el.threads.replaceChildren(node("li", "hint", "No conversations yet."));
    return;
  }

  el.threads.replaceChildren(
    ...threads.map((thread) => {
      const button = node("button", "thread", thread.title);
      if (thread.id === activeThreadId) button.classList.add("is-active");
      button.title = thread.title;
      button.onclick = () => openThread(thread.id);

      const item = document.createElement("li");
      item.append(button);
      return item;
    }),
  );
}

function drawDocuments() {
  if (!documents.length) {
    el.documents.replaceChildren();
    el.docHint.textContent = "No documents yet. Add a PDF to get started.";
    drawScope();
    return;
  }

  el.documents.replaceChildren(
    ...documents.map((document_) => {
      const button = node("button", "doc");
      if (scope.has(document_.filename)) button.classList.add("is-picked");
      button.title = document_.filename;
      button.append(
        node("span", "doc__name", document_.filename),
        node("span", "doc__count", document_.chunks),
      );
      button.onclick = () => {
        // Narrowing to one document removes the cross-document competition for
        // retrieval slots entirely — which is the whole point of the filter.
        if (scope.has(document_.filename)) scope.delete(document_.filename);
        else scope.add(document_.filename);
        drawDocuments();
      };

      const item = document.createElement("li");
      item.append(button);
      return item;
    }),
  );

  el.docHint.textContent = "Click a document to search only in it.";
  drawScope();
}

function drawScope() {
  const picked = [...scope];
  el.scope.hidden = picked.length === 0;
  if (!picked.length) return;

  el.scope.replaceChildren(
    document.createTextNode("Searching in "),
    node("b", null, picked.join(", ")),
  );
}

function setNote(message, bad = false) {
  el.note.hidden = !message;
  el.note.textContent = message;
  el.note.classList.toggle("is-bad", bad);
}

/* ── Stream ───────────────────────────────────────────────────────────── */

function citations(sources) {
  if (!sources || !sources.length) return null;

  const list = node("ul", "cites");
  const seen = new Set();

  for (const source of sources) {
    // The same page often backs several retrieved chunks; listing it once is
    // enough for the reader.
    const key = `${source.filename}#${source.page}`;
    if (seen.has(key)) continue;
    seen.add(key);

    const item = document.createElement("li");
    item.append(
      node("span", null, source.filename),
      node("em", null, `p. ${source.page}`),
    );
    list.append(item);
  }

  return list;
}

function answerNode(message) {
  const wrapper = document.createElement("div");
  const prose = node("div", "prose");
  prose.innerHTML = renderMarkdown(message.content);
  wrapper.append(prose);

  const cites = citations(message.sources);
  if (cites) wrapper.append(cites);

  return wrapper;
}

function drawMessages(messages) {
  const turns = [];
  let turn = null;

  for (const message of messages) {
    // Every question opens a new turn; the answer that follows joins it.
    if (message.role === "user" || !turn) {
      turn = node("article", "turn");
      turns.push(turn);
    }
    turn.append(
      message.role === "user"
        ? node("p", "ask", message.content)
        : answerNode(message),
    );
  }

  el.streamInner.replaceChildren(...turns);
  scrollToEnd();
}

function drawEmptyState() {
  const block = node("div", "empty");
  block.append(
    node("h2", "empty__title", "Ask your documents anything."),
    node("div", "empty__rule"),
    node(
      "p",
      "empty__text",
      documents.length
        ? `${documents.length} document${documents.length > 1 ? "s" : ""} indexed and ready. Answers come back with the page they were found on.`
        : "No documents indexed yet. Add a PDF from the panel on the left, then ask away.",
    ),
  );

  const keys = node("p", "empty__text");
  keys.append(
    node("kbd", null, "Enter"),
    document.createTextNode(" to send  ·  "),
    node("kbd", null, "Shift"),
    document.createTextNode(" + "),
    node("kbd", null, "Enter"),
    document.createTextNode(" for a new line"),
  );
  block.append(keys);

  el.streamInner.replaceChildren(block);
}

function showOffline(message) {
  const block = node("div", "offline");
  block.append(
    node("strong", null, "Backend not reachable."),
    document.createTextNode(` Is uvicorn running on port 8001? (${message})`),
  );
  el.streamInner.prepend(block);
}

function scrollToEnd() {
  el.stream.scrollTop = el.stream.scrollHeight;
}

/* ── Actions ──────────────────────────────────────────────────────────── */

async function loadThreads() {
  threads = await request("/threads");
  drawThreads();
}

async function loadDocuments() {
  documents = await request("/documents");
  drawDocuments();
}

async function loadMessages() {
  if (activeThreadId === null) {
    drawEmptyState();
    return;
  }
  drawMessages(await request(`/threads/${activeThreadId}/messages`));
}

async function openThread(id) {
  activeThreadId = id;
  drawThreads();
  await loadMessages();
  el.question.focus();
}

function startNewChat() {
  // Nothing is written to the database — the row appears on the first question.
  activeThreadId = null;
  drawThreads();
  drawEmptyState();
  el.question.focus();
}

async function ask(question) {
  asking = true;
  setSending(true);

  try {
    // A brand new chat has no row yet. Create it now, named after this first
    // question so the rail entry is readable.
    if (activeThreadId === null) {
      const title = question.length <= 40 ? question : `${question.slice(0, 40)}…`;
      const created = await request("/threads", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      });
      activeThreadId = created.id;
      await loadThreads();
      el.streamInner.replaceChildren();
    }

    // Put the question on screen straight away, so nobody is staring at an
    // unchanged page while the answer is being generated.
    const turn = node("article", "turn");
    const thinking = node("div", "thinking");
    thinking.append(node("i"), node("i"), node("i"));
    turn.append(node("p", "ask", question), thinking);
    el.streamInner.append(turn);
    scrollToEnd();

    try {
      await request("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          sources: [...scope],
          thread_id: activeThreadId,
        }),
      });
      // The backend saved both messages, so read them back instead of keeping
      // a second copy of the conversation here.
      await loadMessages();
    } catch (error) {
      thinking.replaceWith(node("p", "failure", error.message));
      scrollToEnd();
    }
  } catch (error) {
    showOffline(error.message);
  } finally {
    asking = false;
    setSending(false);
    el.question.focus();
  }
}

async function upload(file) {
  const form = new FormData();
  // No Content-Type header here on purpose — the browser has to set it itself
  // so the multipart boundary matches the body.
  form.append("file", file, file.name);

  setNote(`Ingesting ${file.name}…`);
  try {
    const result = await request("/upload", { method: "POST", body: form });
    let message = `${result.filename} — ${result.chunks} chunks`;
    if (result.replaced) message += `, replaced ${result.replaced} older ones`;
    setNote(message);
    await loadDocuments();
    if (activeThreadId === null) drawEmptyState();
  } catch (error) {
    setNote(`Upload failed: ${error.message}`, true);
  }
}

/* ── Composer wiring ──────────────────────────────────────────────────── */

function autoGrow() {
  el.question.style.height = "auto";
  el.question.style.height = `${el.question.scrollHeight}px`;
}

function setSending(sending) {
  el.send.disabled = sending || !el.question.value.trim();
  el.send.textContent = sending ? "…" : "Ask";
}

el.question.addEventListener("input", () => {
  autoGrow();
  setSending(asking);
});

el.question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    el.composer.requestSubmit();
  }
});

el.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = el.question.value.trim();
  if (!question || asking) return;

  el.question.value = "";
  autoGrow();
  ask(question);
});

el.newChat.onclick = startNewChat;
el.addPdf.onclick = () => el.file.click();

el.file.onchange = () => {
  const [file] = el.file.files;
  // Reset first, so picking the same file twice still fires a change event.
  el.file.value = "";
  if (file) upload(file);
};

/* ── Boot ─────────────────────────────────────────────────────────────── */

(async () => {
  drawEmptyState();
  try {
    await Promise.all([loadThreads(), loadDocuments()]);
    drawEmptyState();
  } catch (error) {
    showOffline(error.message);
  }
})();
