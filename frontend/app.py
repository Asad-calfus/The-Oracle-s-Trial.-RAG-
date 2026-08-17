import requests
import streamlit as st

# Change this if your backend is running on a different port.
API_BASE_URL = "http://127.0.0.1:8001"

st.set_page_config(page_title="SmartDoc", page_icon="📄")
st.title("SmartDoc — Document Q&A")

# Chat history now lives in Postgres, not in the browser — so the only thing
# session_state still has to remember is which thread is open. None means
# "a new chat that hasn't been saved yet".
if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = None


def render_sources(sources):
    """Show an answer's citations underneath it."""
    if not sources:
        return
    st.caption("Sources:")
    for source in sources:
        st.caption(f"- {source['filename']} — Page {source['page']}")


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


# The sidebar is rendered BEFORE the chat area on purpose: switching threads
# happens here, and the chat below needs to already know which one to draw.
with st.sidebar:
    if st.button("➕ New chat", use_container_width=True):
        # Nothing is written to the database yet. The thread row is created
        # only when a first question actually arrives, so empty chats never
        # pile up in the sidebar.
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

    st.header("Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    if uploaded_file is not None and st.button("Upload"):
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
        }
        with st.spinner("Ingesting..."):
            response = requests.post(f"{API_BASE_URL}/upload", files=files)
        if response.ok:
            data = response.json()
            message = f"Ingested {data['filename']} into {data['chunks']} chunks."
            if data.get("replaced"):
                message += f" Replaced {data['replaced']} chunks from a previous upload."
            st.success(message)
        else:
            st.error(f"Upload failed: {response.text}")

    # Rendered AFTER the upload block so a just-uploaded file shows up
    # immediately, rather than one interaction late.
    st.header("Documents in use")
    selected_sources = []
    documents = fetch("/documents")

    if documents is None:
        st.warning("Backend not reachable. Is uvicorn running?")
    elif not documents:
        st.caption("No documents uploaded yet.")
    else:
        for document in documents:
            st.caption(f"- {document['filename']} ({document['chunks']} chunks)")

        # Narrowing the search to one document removes the cross-document
        # competition for retrieval slots entirely. Empty = search everything.
        selected_sources = st.multiselect(
            "Search only in",
            [document["filename"] for document in documents],
            placeholder="All documents",
        )

# Read the active thread AFTER the sidebar, since the sidebar is what changes it.
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
    # A brand new chat has no row yet — create it now, named after this first
    # question so the sidebar entry is readable.
    if active_thread_id is None:
        title = question if len(question) <= 40 else question[:40] + "…"
        created = requests.post(f"{API_BASE_URL}/threads", json={"title": title})
        active_thread_id = created.json()["id"]
        st.session_state.active_thread_id = active_thread_id

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
