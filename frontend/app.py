import requests
import streamlit as st

# Change this if your backend is running on a different port.
API_BASE_URL = "http://127.0.0.1:8001"

st.set_page_config(page_title="SmartDoc", page_icon="📄")
st.title("SmartDoc — Document Q&A")

# Streamlit re-runs this whole script top-to-bottom on every interaction, so
# normal variables are wiped each time. session_state is the only thing that
# survives a rerun — which makes it the only place chat history can live.
if "threads" not in st.session_state:
    st.session_state.threads = {}
if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = None
if "next_thread_id" not in st.session_state:
    st.session_state.next_thread_id = 1


def create_thread():
    """Open a new empty chat and make it the active one.

    If the current thread is still empty, we just stay on it — otherwise
    clicking "New chat" repeatedly would stack up identical blank threads.
    """
    active_id = st.session_state.active_thread_id
    if active_id is not None and not st.session_state.threads[active_id]["messages"]:
        return

    thread_id = st.session_state.next_thread_id
    st.session_state.next_thread_id += 1
    st.session_state.threads[thread_id] = {"title": "New chat", "messages": []}
    st.session_state.active_thread_id = thread_id


def render_sources(sources):
    """Show an answer's citations underneath it."""
    if not sources:
        return
    st.caption("Sources:")
    for source in sources:
        st.caption(f"- {source['filename']} — Page {source['page']}")


# There must always be one thread open for the user to type into.
if st.session_state.active_thread_id is None:
    create_thread()

# The sidebar is rendered BEFORE the chat area on purpose: switching threads
# happens here, and the chat below needs to already know which one to draw.
with st.sidebar:
    if st.button("➕ New chat", use_container_width=True):
        create_thread()

    for thread_id, thread in reversed(list(st.session_state.threads.items())):
        is_active = thread_id == st.session_state.active_thread_id
        # key= is required because several buttons can share the same label.
        if st.button(
            thread["title"],
            key=f"thread-{thread_id}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.active_thread_id = thread_id
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
    try:
        documents_response = requests.get(f"{API_BASE_URL}/documents")
        documents = documents_response.json() if documents_response.ok else []
    except requests.RequestException:
        # Unlike upload/query, this call runs on EVERY rerun — so a backend
        # that isn't running would otherwise crash the whole page on load.
        documents = None

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

# Mutating this dict mutates session_state directly — it's a reference, not a copy.
active_thread = st.session_state.threads[st.session_state.active_thread_id]

# Redraw the whole conversation from scratch on every rerun — Streamlit only
# shows what this script renders right now, so past messages have to be
# replayed explicitly rather than staying on screen by themselves.
for message in active_thread["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

# chat_input pins itself to the bottom of the page and returns the submitted
# text once, then None on the reruns that follow.
question = st.chat_input("Ask a question about your documents")
if question:
    # Name the thread after its first question so the sidebar is readable.
    if not active_thread["messages"]:
        active_thread["title"] = (
            question if len(question) <= 40 else question[:40] + "…"
        )

    active_thread["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{API_BASE_URL}/query",
                json={"question": question, "sources": selected_sources},
            )

        if response.ok:
            data = response.json()
            st.write(data["answer"])
            render_sources(data["sources"])
            active_thread["messages"].append(
                {
                    "role": "assistant",
                    "content": data["answer"],
                    "sources": data["sources"],
                }
            )
        else:
            error_message = f"Query failed: {response.text}"
            st.error(error_message)
            active_thread["messages"].append(
                {"role": "assistant", "content": error_message, "sources": []}
            )
