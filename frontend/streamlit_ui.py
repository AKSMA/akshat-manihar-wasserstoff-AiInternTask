import streamlit as st
import requests
import pandas as pd
import os
import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"

API_URL = "http://localhost:8000/api"
for _ in range(10):
    try:
        r = requests.get(f"{API_URL}/ping")  # create a dummy ping endpoint
        if r.status_code == 200:
            break
    except requests.exceptions.ConnectionError:
        time.sleep(1)
else:
    st.error("Backend API not reachable.") # Update to cloud URL if deployed

st.set_page_config(page_title="Wasserstoff Gen-AI", layout="wide")
st.title("📄 Wasserstoff Document Q&A System")

# --- Sidebar: Upload Documents ---
st.sidebar.header("Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload up to 75 files", type=["pdf", "jpg", "jpeg", "png", "txt", "docx"], accept_multiple_files=True
)

if st.sidebar.button("Upload"):
    if not uploaded_files:
        st.warning("Please select at least one file.")
    elif len(uploaded_files) > 75:
        st.error("You can upload a maximum of 75 files.")
    else:
        with st.spinner("Uploading files..."):
            files = [("files", (f.name, f, f.type)) for f in uploaded_files]
            r = requests.post(f"{API_URL}/upload/", files=files)
            if r.ok:
                st.success(f"✅ Uploaded {len(uploaded_files)} files successfully!")
            else:
                st.error("❌ Upload failed.")

# --- Sidebar: View Uploaded Documents ---
st.sidebar.markdown("### 📁 Uploaded Documents")
try:
    response = requests.get(f"{API_URL}/documents/")
    if response.ok:
        doc_list = response.json().get("documents", [])
        if doc_list:
            for doc in doc_list:
                st.sidebar.markdown(f"📄 `{doc['filename']}`")
        else:
            st.sidebar.info("No documents uploaded yet.")
    else:
        st.sidebar.error("Couldn't fetch document list.")
except Exception as e:
    st.sidebar.error(f"Error: {e}")

# --- Main: Question Input ---
st.subheader("Ask a question about your uploaded documents")
query = st.text_input("Enter your question:")

if st.button("Submit Query"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            # 1. Query for direct answer and citations
            r = requests.post(f"{API_URL}/query_chain", json={"question": query})
            if r.ok:
                data = r.json()
                st.markdown("### 🤖 Synthesized Answer")
                st.success(data["answer"])

                st.markdown("### 📚 Supporting Citations")
                if data["citations"]:
                    df = pd.DataFrame(data["citations"])
                    df.index = df.index + 1
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No citations returned.")
            else:
                st.error("Failed to get answer.")

            # 2. Query for theme synthesis
            synth = requests.post(f"{API_URL}/synthesize", json={"question": query})
            if synth.ok:
                st.markdown("### 🧠 Synthesized Themes")
                themes = synth.json().get("themes", [])
                if themes:
                    for idx, theme in enumerate(themes, 1):
                        st.markdown(f"**Theme {idx} – {theme['title']}**")
                        st.info(theme["summary"])
                        if theme["documents"]:
                            y=[]
                            for d in theme["documents"]:
                                data=d.replace('.','').replace('(','').replace(')','').split('#')
                                x={"DocID":data[0].strip(), "Page":data[1].strip() if data[1].strip() else None, "Chunk":data[2].strip()}
                                y.append(x)
                            df = pd.DataFrame(y)
                            df.index = df.index + 1
                            st.dataframe(df, use_container_width=True)
                        st.markdown("---")
                else:
                    st.info("No themes identified.")
            else:
                st.error("Failed to generate themes.")
