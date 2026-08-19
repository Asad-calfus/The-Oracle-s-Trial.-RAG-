import json

import requests
import streamlit as st

# Change this if your backend is running on a different port.
API_BASE_URL = "http://127.0.0.1:8001"

st.set_page_config(page_title="SmartDoc", page_icon="📄")
st.title("SmartDoc — Document Q&A")

# Chat history now lives in Postgres, not in the browser — so the only thing
# session_state still has to remember is which thread is open. None means
# "a new chat that hasn't been saved yet" — it has no row in the database
# and no documents of its own until something creates it (see ensure_thread).
if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = None

# Holds the last-generated graph's HTML (or an error), so it can be
# rendered in the main area below the sidebar, outside the narrow sidebar
# column — a force-directed graph needs real width to be readable.
if "graph_html" not in st.session_state:
    st.session_state.graph_html = None
if "graph_error" not in st.session_state:
    st.session_state.graph_error = None


def render_sources(sources):
    """Show an answer's citations underneath it, each with a quoted excerpt.

    Not every format has a page (Section 20.2 — .txt/.md/.docx have no
    real pagination), so the "— Page N" segment only appears when a page
    actually exists, instead of literally printing "— Page None".
    """
    if not sources:
        return
    st.caption("Sources:")
    for source in sources:
        page = source.get("page")
        location = f"{source['filename']} — Page {page}" if page is not None else source["filename"]
        st.caption(f"- {location}")
        excerpt = source.get("excerpt")
        if excerpt:
            st.caption(f"   _\"{excerpt}\"_")


def render_thinking(thinking, key):
    """Collapsed-by-default panel showing the pipeline's own real numbers —
    not a separate LLM-generated explanation (PLANNING.md 19.2). Nothing
    here is shown unless the user opens it.
    """
    if not thinking:
        return
    with st.expander("🧠 Show thinking", expanded=False):
        if thinking.get("rewritten_question"):
            st.caption(f"Rewritten question: _{thinking['rewritten_question']}_")

        source_filter = thinking.get("source_filter") or {}
        filter_bits = [f"thread_id={source_filter.get('thread_id')}"]
        if source_filter.get("filenames"):
            filter_bits.append(f"filenames={source_filter['filenames']}")
        st.caption(f"Search scope: {', '.join(filter_bits)}")

        # All 5 tunable knobs (Section 21) actually used for THIS answer —
        # config defaults unless this thread has its own overrides.
        settings_used = thinking.get("settings_used") or {}
        st.caption(
            f"Retrieved {thinking.get('retrieved_count', 0)} candidate chunk(s) — "
            f"similarity threshold: {settings_used.get('similarity_threshold')} "
            "(lower score = more similar)"
        )

        # Per-chunk breakdown: exactly which chunks were involved and what
        # happened to each, not just aggregate counts.
        breakdown = thinking.get("chunk_breakdown") or []
        if breakdown:
            st.dataframe(breakdown, use_container_width=True, hide_index=True, key=f"{key}-breakdown")

        st.caption(
            f"{thinking.get('passed_threshold_count', 0)} passed the threshold, "
            f"{thinking.get('kept_after_rerank_count', 0)} kept after reranking "
            f"(retrieval_top_k={settings_used.get('retrieval_top_k')}, "
            f"rerank_candidate_k={settings_used.get('rerank_candidate_k')})"
        )

        st.caption(
            f"Model: {thinking.get('model')} "
            f"(temperature={settings_used.get('llm_temperature')}, "
            f"rewrite_history_messages={settings_used.get('rewrite_history_messages')})"
        )

        usage = thinking.get("token_usage")
        if usage:
            st.caption(
                f"Token usage — input: {usage.get('input_tokens')}, "
                f"output: {usage.get('output_tokens')}, total: {usage.get('total_tokens')}"
            )


def fetch(path):
    """GET a backend endpoint, returning None if the backend isn't reachable.

    These run on EVERY rerun, so an unreachable backend would otherwise crash
    the page on load rather than just failing the one action that needed it.
    """
    try:
        response = requests.get(f"{API_BASE_URL}{path}")
        return response.json() if response.ok else None
    except requests.RequestException:
        return None


def ensure_thread(title):
    """Create a thread if none is active yet, and return its id either way.

    A brand new chat has no thread row until SOMETHING creates one — and now
    that can be either the first upload or the first question, whichever
    happens first. Both call this instead of duplicating the creation logic.
    """
    if st.session_state.active_thread_id is None:
        created = requests.post(f"{API_BASE_URL}/threads", json={"title": title})
        st.session_state.active_thread_id = created.json()["id"]
    return st.session_state.active_thread_id


# The sidebar is rendered BEFORE the chat area on purpose: switching threads
# happens here, and the chat below needs to already know which one to draw.
with st.sidebar:
    if st.button("➕ New chat", use_container_width=True):
        # Nothing is written to the database yet — this thread is created
        # lazily by ensure_thread(), on whichever happens first: an upload
        # or a question.
        st.session_state.active_thread_id = None
        # A graph generated for the previous chat's document has no business
        # showing up under a brand new, document-less chat.
        st.session_state.graph_html = None
        st.session_state.graph_error = None

    for thread in fetch("/threads") or []:
        is_active = thread["id"] == st.session_state.active_thread_id
        # key= is required because several buttons can share the same label.
        if st.button(
            thread["title"],
            key=f"thread-{thread['id']}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.active_thread_id = thread["id"]
            # A graph belongs to the thread/document it was generated for —
            # switching threads shouldn't carry it along.
            st.session_state.graph_html = None
            st.session_state.graph_error = None
            # Buttons above this one in the loop were already drawn using the
            # OLD active id, so their highlighting is stale — rerun to redraw
            # the whole sidebar consistently.
            st.rerun()

    st.divider()

    st.header("Upload a document")
    uploaded_file = st.file_uploader(
        "Choose a document", type=["pdf", "docx", "txt", "md"]
    )
    include_images = st.checkbox(
        "Describe images (charts/diagrams/scanned pages) using AI",
        help="PDF only. Uses a vision AI call per image-containing page, "
        "so it costs a little extra and takes longer. Leave unchecked for "
        "plain-text PDFs, or when uploading a .docx/.txt/.md file.",
    )
    if uploaded_file is not None and st.button("Upload"):
        # This chat's own document, tagged with this chat's own thread —
        # that tag is what keeps it invisible to every other chat's search.
        thread_id = ensure_thread(f"Chat about {uploaded_file.name}")

        # uploaded_file.type is whatever the browser reported — correct
        # per-format, unlike a single hardcoded "application/pdf" would be
        # now that more than one file type is accepted.
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }
        spinner_text = "Ingesting (describing images)..." if include_images else "Ingesting..."
        with st.spinner(spinner_text):
            response = requests.post(
                f"{API_BASE_URL}/upload",
                files=files,
                data={"thread_id": thread_id, "include_images": include_images},
            )
        if response.ok:
            data = response.json()
            message = f"Ingested {data['filename']} into {data['chunks']} chunks."
            if data.get("replaced"):
                message += f" Replaced {data['replaced']} chunks from a previous upload."
            st.success(message)
            # Refreshes the sidebar's thread list (if this upload just
            # created one) and the document list section below.
            st.rerun()
        else:
            st.error(f"Upload failed: {response.text}")

    # Rendered AFTER the upload block so a just-uploaded file shows up
    # immediately, rather than one interaction late.
    st.header("Documents in use")
    selected_sources = []
    active_thread_id = st.session_state.active_thread_id

    if active_thread_id is None:
        # A brand new chat has no thread yet, so it can't have documents
        # yet either — nothing to fetch.
        st.caption("Upload a document above to start this chat.")
    else:
        documents = fetch(f"/documents?thread_id={active_thread_id}")

        if documents is None:
            st.warning("Backend not reachable. Is uvicorn running?")
        elif not documents:
            st.caption("No documents uploaded yet.")
        else:
            for document in documents:
                st.caption(f"- {document['filename']} ({document['chunks']} chunks)")
                if st.button(
                    "🕸️ Knowledge graph",
                    key=f"graph-{document['filename']}",
                    help="Uses AI to extract entities/relationships from this "
                    "document. The FIRST click can take 1-2 minutes and makes "
                    "several AI calls; after that it's cached and instant. "
                    "Refused for very large documents.",
                ):
                    with st.spinner("Building knowledge graph (first time only)..."):
                        graph_response = requests.post(
                            f"{API_BASE_URL}/documents/graph",
                            params={
                                "filename": document["filename"],
                                "thread_id": active_thread_id,
                            },
                        )
                    if graph_response.ok:
                        st.session_state.graph_html = graph_response.json()["html"]
                        st.session_state.graph_error = None
                    else:
                        st.session_state.graph_html = None
                        try:
                            detail = graph_response.json().get("detail", graph_response.text)
                        except ValueError:
                            detail = graph_response.text
                        st.session_state.graph_error = detail
                    st.rerun()

            # Narrowing further to specific files within THIS thread's own
            # documents — the thread boundary itself is already enforced
            # server-side, this is an additional, optional narrowing.
            selected_sources = st.multiselect(
                "Search only in",
                [document["filename"] for document in documents],
                placeholder="All documents",
            )

    # Settings are per-thread (Section 21), so there's nothing to show or
    # save until a thread actually exists.
    if active_thread_id is not None:
        defaults = fetch("/config/defaults") or {}
        current_thread = next(
            (t for t in fetch("/threads") or [] if t["id"] == active_thread_id), None
        )
        thread_overrides = (current_thread or {}).get("settings") or {}
        resolved = {**defaults, **thread_overrides}

        def sync_setting(key, new_value):
            """Persist immediately (PATCH sends just this one key) only
            when the slider actually moved — every OTHER widget interaction
            on the page also reruns this script, and on those reruns the
            slider simply returns its already-stored value, so this stays
            a no-op the rest of the time."""
            if new_value != resolved.get(key):
                requests.patch(
                    f"{API_BASE_URL}/threads/{active_thread_id}/settings",
                    json={key: new_value},
                )
                st.rerun()

        with st.expander("⚙️ Advanced settings"):
            similarity_threshold = st.slider(
                "Similarity threshold",
                min_value=0.5, max_value=3.0, step=0.1,
                value=float(resolved.get("similarity_threshold", 1.8)),
                help="Lower = stricter (only very close matches count, more "
                "'I don't know'). Higher = looser (more chunks reach the AI).",
            )
            sync_setting("similarity_threshold", similarity_threshold)

            retrieval_top_k = st.slider(
                "Sources to consider (after reranking)",
                min_value=1, max_value=20, step=1,
                value=int(resolved.get("retrieval_top_k", 6)),
                help="How many chunks reach the final answer. More = more "
                "context but more tokens/cost.",
            )
            sync_setting("retrieval_top_k", retrieval_top_k)

            rerank_candidate_k = st.slider(
                "Initial candidate pool (before reranking)",
                min_value=5, max_value=30, step=1,
                value=int(resolved.get("rerank_candidate_k", 15)),
                help="How wide the initial search is before narrowing down. "
                "Higher = less likely to miss something relevant, slightly slower.",
            )
            sync_setting("rerank_candidate_k", rerank_candidate_k)

            rewrite_history_messages = st.slider(
                "Conversation memory depth",
                min_value=0, max_value=20, step=1,
                value=int(resolved.get("rewrite_history_messages", 6)),
                help="How many past messages are used to resolve follow-up "
                "questions ('his role?' etc). 0 turns conversational memory off.",
            )
            sync_setting("rewrite_history_messages", rewrite_history_messages)

            llm_temperature = st.slider(
                "Answer creativity (temperature)",
                min_value=0.0, max_value=1.0, step=0.1,
                value=float(resolved.get("llm_temperature", 0)),
                help="0 is the safest default for fact-grounded answers — "
                "raising this trades consistency for varied phrasing.",
            )
            sync_setting("llm_temperature", llm_temperature)
            if llm_temperature > 0:
                st.caption(
                    "⚠️ Non-zero temperature means the same question can get "
                    "differently-worded answers on different runs."
                )

            st.divider()
            if thread_overrides and st.button("↩️ Reset to defaults", use_container_width=True):
                requests.delete(f"{API_BASE_URL}/threads/{active_thread_id}/settings")
                st.rerun()

# Read again here: the sidebar above may have just created a thread via an
# upload, and the chat area below needs that fresh value, not a stale one
# read before the sidebar ran.
active_thread_id = st.session_state.active_thread_id

# Rendered here (full page width), not inside the sidebar — a
# force-directed graph needs real space to be readable.
if st.session_state.graph_error:
    st.error(f"Couldn't build knowledge graph: {st.session_state.graph_error}")
if st.session_state.graph_html:
    st.subheader("Knowledge Graph")
    st.components.v1.html(st.session_state.graph_html, height=600, scrolling=True)

# Replay the conversation from the database. Streamlit only shows what this
# script renders right now, so past messages have to be drawn explicitly
# rather than staying on screen by themselves.
if active_thread_id is not None:
    for index, message in enumerate(fetch(f"/threads/{active_thread_id}/messages") or []):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                render_sources(message["sources"] or [])
                render_thinking(message.get("thinking"), key=f"thinking-{index}")

# chat_input pins itself to the bottom of the page and returns the submitted
# text once, then None on the reruns that follow.
question = st.chat_input("Ask a question about your documents")
if question:
    title = question if len(question) <= 40 else question[:40] + "…"
    active_thread_id = ensure_thread(title)

    # Show the question straight away so the user isn't staring at a blank
    # screen while the answer is being generated.
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer_so_far = ""
        final_payload = None

        try:
            with requests.post(
                f"{API_BASE_URL}/query/stream",
                json={
                    "question": question,
                    "sources": selected_sources,
                    "thread_id": active_thread_id,
                },
                stream=True,
            ) as response:
                if response.ok:
                    for line in response.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        event = json.loads(line)
                        if event["type"] == "token":
                            answer_so_far += event["text"]
                            # A blinking cursor while tokens are still arriving,
                            # so a mid-sentence render doesn't look "finished".
                            placeholder.write(answer_so_far + " ▌")
                        else:
                            final_payload = event

                    if final_payload is not None:
                        placeholder.write(final_payload["answer"])
                        render_sources(final_payload.get("sources") or [])
                        render_thinking(final_payload.get("thinking"), key="thinking-live")
                else:
                    st.error(f"Query failed: {response.text}")
        except requests.RequestException as e:
            st.error(f"Backend not reachable: {e}")

    # Both messages are already saved by the backend by the time the stream
    # ends — rerun so the sidebar/thread list stay in sync (e.g. a brand new
    # chat's title now exists), without keeping a second copy of the answer
    # here.
    if final_payload is not None:
        st.rerun()
