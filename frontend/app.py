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

            # Narrowing further to specific files within THIS thread's own
            # documents — the thread boundary itself is already enforced
            # server-side, this is an additional, optional narrowing.
            selected_sources = st.multiselect(
                "Search only in",
                [document["filename"] for document in documents],
                placeholder="All documents",
            )

# Read again here: the sidebar above may have just created a thread via an
# upload, and the chat area below needs that fresh value, not a stale one
# read before the sidebar ran.
active_thread_id = st.session_state.active_thread_id

# Replay the conversation from the database. Streamlit only shows what this
# script renders right now, so past messages have to be drawn explicitly
# rather than staying on screen by themselves.
if active_thread_id is not None:
    for message in fetch(f"/threads/{active_thread_id}/messages") or []:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                render_sources(message["sources"] or [])

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
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{API_BASE_URL}/query",
                json={
                    "question": question,
                    "sources": selected_sources,
                    "thread_id": active_thread_id,
                },
            )

        if response.ok:
            # The backend already saved both messages, so rerun and let the
            # replay loop above read them back — no second copy kept here.
            st.rerun()
        else:
            st.error(f"Query failed: {response.text}")
