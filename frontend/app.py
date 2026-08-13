import requests
import streamlit as st

# Change this if your backend is running on a different port.
API_BASE_URL = "http://127.0.0.1:8001"

st.title("SmartDoc — Document Q&A")

st.header("Upload a PDF")
uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
if uploaded_file is not None and st.button("Upload"):
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
    response = requests.post(f"{API_BASE_URL}/upload", files=files)
    if response.ok:
        data = response.json()
        st.success(f"Ingested {data['filename']} into {data['chunks']} chunks.")
    else:
        st.error(f"Upload failed: {response.text}")

st.header("Ask a question")
question = st.text_input("Your question")
if question and st.button("Ask"):
    response = requests.post(f"{API_BASE_URL}/query", json={"question": question})
    if response.ok:
        data = response.json()
        st.write(data["answer"])
        if data["sources"]:
            st.caption("Sources:")
            for source in data["sources"]:
                st.caption(f"- {source['filename']} — Page {source['page']}")
    else:
        st.error(f"Query failed: {response.text}")
