# 🧠 Wasserstoff Gen-AI RAG Chatbot

This project is a full-stack Retrieval-Augmented Generation (RAG) system developed for the **Wasserstoff Gen-AI Internship Task**. It enables users to:

- Upload documents (PDF, scanned images, DOCX, TXT)
- Extract and embed their contents with OCR + FAISS
- Ask natural language questions about the uploaded data
- Get LLM-generated answers with proper citations (DocID, Page, Chunk)
- View synthesized themes extracted from related documents

---

## 🔧 Installation

### 1. Clone the repo
```bash
git clone https://github.com/yourname/chatbot_theme_identifier.git
cd chatbot_theme_identifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start backend (FastAPI)
```bash
pyhton backend/api/main.py
```

### 4. Start frontend (Streamlit)
```bash
cd ../frontend
streamlit run streamlit_ui.py
```

Make sure `API_URL` in `streamlit_ui.py` matches your backend URL.

---

## 📦 API Endpoints

- `POST /api/upload/`: Upload up to 75 documents
- `GET /api/documents/`: List uploaded files
- `POST /api/query_chain`: Ask a question and get an answer + citations
- `POST /api/synthesize`: Get synthesized themes across documents

---

## ✨ Features

- 🔍 OCR for scanned images
- 🧠 GPT-based query + synthesis
- 📚 FAISS vector search
- 🧾 Citations: DocID, Page, Chunk
- 📊 Streamlit UI with answer + themes

---

## 🧪 Sample Use
1. Upload your contracts, policies, reports
2. Ask: "What penalties are discussed in these documents?"
3. Get a structured answer + document references
4. See themes like "Regulatory non-compliance" and "Late payment penalties"

---

## 📜 License
MIT


