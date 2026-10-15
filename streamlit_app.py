import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="DocuMind AI", page_icon="📚", layout="wide")
st.markdown("""
<style>
    .stApp { background: #f5f4ef; }
    [data-testid="stSidebar"] { background: #173b35; color: white; }
    [data-testid="stSidebar"] * { color: white; }
    .source { padding: 1rem; border: 1px solid #d8d3c5; border-radius: 10px; background: white; margin-bottom: .75rem; }
</style>
""", unsafe_allow_html=True)


def api_error(response):
    try:
        return response.json().get("detail", "Something went wrong.")
    except ValueError:
        return "The API is unavailable. Make sure the backend is running."


with st.sidebar:
    st.title("DocuMind AI")
    st.caption("Your evidence-first document assistant")
    st.subheader("Add a document")
    upload = st.file_uploader("Upload a text-based PDF", type=["pdf"])
    if upload and st.button("Process document", type="primary", use_container_width=True):
        with st.spinner("Reading and indexing…"):
            try:
                response = requests.post(
                    f"{API_URL}/documents/upload",
                    files={"file": (upload.name, upload.getvalue(), "application/pdf")},
                    timeout=120,
                )
                if response.ok:
                    st.success(f"Indexed {upload.name}")
                    st.rerun()
                else:
                    st.error(api_error(response))
            except requests.RequestException:
                st.error("Cannot reach the API. Start the backend and try again.")

    st.divider()
    st.subheader("Your documents")
    try:
        docs_response = requests.get(f"{API_URL}/documents", timeout=10)
        documents = docs_response.json() if docs_response.ok else []
    except requests.RequestException:
        documents = []
    if not documents:
        st.caption("No indexed documents yet.")
    for document in documents:
        left, right = st.columns([4, 1])
        left.markdown(f"**{document['filename']}**  \n{document['pages']} pages")
        if right.button("×", key=document["id"], help="Delete document"):
            requests.delete(f"{API_URL}/documents/{document['id']}", timeout=10)
            st.rerun()

st.title("Ask your documents")
st.write("Get concise answers grounded in the PDFs you upload, with page-level evidence.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for source in message.get("sources", []):
            st.markdown(
                f"<div class='source'><b>[{source['reference']}] {source['filename']}</b> · page {source['page']}<br>"
                f"<small>{source['excerpt']}</small></div>", unsafe_allow_html=True
            )

question = st.chat_input("Ask a question about your documents…", disabled=not documents)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching for evidence…"):
            try:
                response = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=60)
                if response.ok:
                    result = response.json()
                    st.markdown(result["answer"])
                    for source in result["sources"]:
                        st.markdown(
                            f"<div class='source'><b>[{source['reference']}] {source['filename']}</b> · page {source['page']}<br>"
                            f"<small>{source['excerpt']}</small></div>", unsafe_allow_html=True
                        )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
                    )
                else:
                    st.error(api_error(response))
            except requests.RequestException:
                st.error("Cannot reach the API. Please try again.")

