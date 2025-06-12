from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import shutil
import requests
import uvicorn
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import (
    UnstructuredPDFLoader, UnstructuredImageLoader,
    TextLoader, UnstructuredWordDocumentLoader, PyPDFLoader
)
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.chains import create_retrieval_chain
import os
from dotenv import load_dotenv 
load_dotenv()

groq_api_key = os.environ['GROQ_API_KEY']
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
INDEX_DIR = "faiss_index"

# ======================== MODELS ==============================


class QueryRequest(BaseModel):
    question: str


class DocAnswer(BaseModel):
    DocID: str
    Page: Optional[int]
    ChunkIndex: int
    Excerpt: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[DocAnswer]


class Theme(BaseModel):
    title: str
    documents: List[str]
    summary: str


class ThemeResponse(BaseModel):
    themes: List[Theme]

# ======================== UTILITIES ==============================


def load_single_file(file_path: Path) -> List[Document]:
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        loader = UnstructuredPDFLoader(str(file_path), mode="elements", strategy="ocr_only")
    elif ext in [".jpg", ".jpeg", ".png"]:
        loader = UnstructuredImageLoader(str(file_path))
    elif ext == ".txt":
        loader = TextLoader(str(file_path))
    elif ext == ".docx":
        loader = UnstructuredWordDocumentLoader(str(file_path))
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    docs = loader.load()
    for i, doc in enumerate(docs):
        doc.metadata["DocID"] = file_path.name
        doc.metadata["ChunkIndex"] = i
    return docs


def ingest_documents():
    all_docs = []
    for file_path in UPLOAD_DIR.glob("*"):
        try:
            docs = load_single_file(file_path)
            all_docs.extend(docs)
        except Exception as e:
            print(f"Failed to process {file_path.name}: {e}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000, chunk_overlap=1500)
    chunked_docs = []
    for doc in all_docs:
        chunks = splitter.create_documents([doc.page_content], [doc.metadata])
        chunked_docs.extend(chunks)

    embeddings=HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",      #sentence-transformers/all-MiniLM-l6-v2
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}

)
    vectordb = FAISS.from_documents(chunked_docs, embeddings)
    vectordb.save_local(INDEX_DIR)
    return len(all_docs), len(chunked_docs)

# ======================== ROUTES ==============================


@app.post("/api/upload/")
def upload_documents(files: List[UploadFile] = File(...)):
    if len(files) > 75:
        return JSONResponse(content={"error": "Max 75 files allowed."}, status_code=400)
    for file in files:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    doc_count, chunk_count = ingest_documents()
    return {"message": f"Uploaded {doc_count} documents.", "chunks": chunk_count}


@app.get("/api/documents/")
def list_documents():
    files = [{"filename": f.name} for f in UPLOAD_DIR.glob("*")]
    return {"documents": files}


@app.post("/api/query_chain", response_model=QueryResponse)
def query_chain(payload: QueryRequest):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever()

    system_prompt = """
    You are an AI assistant answering questions based ONLY on the context below.
    Each document snippet includes: DocID, Page, and ChunkIndex.

    - Use only the context.
    - Do not make up facts.
    - Cite using (DocID, Page, Chunk).
    """

    prompt = ChatPromptTemplate.from_template('''
    Answer the following question only based on the context provided.
    Think step by step before providing a detailed answer.
    <context>
    {context}
    </context>                                        
    Question: {input}''')

    llm = llm=ChatGroq(groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",temperature=0)
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": payload.question})
    docs = response["context"]

    citations = []
    for doc in docs:
        citations.append(DocAnswer(
            DocID=doc.metadata.get("DocID", "unknown"),
            Page=doc.metadata.get("page_number"),
            ChunkIndex=doc.metadata.get("ChunkIndex", -1),
            Excerpt=doc.page_content[:300].strip()
        ))

    return QueryResponse(answer=response["answer"], citations=citations)


@app.post("/api/synthesize", response_model=ThemeResponse)
def synthesize_themes(payload: QueryRequest):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever()
    docs = retriever.get_relevant_documents(payload.question)
    chunk_strs = []
    for doc in docs:
        docid = doc.metadata.get("DocID", "unknown")
        page = doc.metadata.get("page_number", "-")
        chunk = doc.metadata.get("ChunkIndex", "-")
        excerpt = doc.page_content.strip().replace("\n", " ")[:400]
        chunk_strs.append(f"{docid} (Page {page}, Chunk {chunk}): {excerpt}")

    chunks_input = "\n".join(chunk_strs)

    prompt_template = PromptTemplate.from_template("""
    You are an AI assistant. Given a question and document excerpts, identify main themes.
    For each theme:
    - Name the theme
    - Give a short summary
    - List supporting citations in format (DocID, Page, Chunk)

    Question:
    {question}

    Document Excerpts:
    {chunks}

    ONLY use this format for your response:
    Theme 1 – [Title]
    Summary: ...
    Citations: ...
    - Cite using (DocID, Page, Chunk).
    - Provide the Citations in a single line only separated by comma while the DocID, Page, Chunk are seperated by hashtag like this (DocID# Page# Chunk).
    """)

    llm = llm=ChatGroq(groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",temperature=0)
    final_prompt = prompt_template.format(
        question=payload.question, chunks=chunks_input)
    result = llm.invoke(final_prompt)
    themes = []
    for block in result.content.split("Theme ")[1:]:
        lines = block.strip().split("\n")
        if len(lines) >= 3:
            title = lines[0].replace("–", "").strip()[2:]
            summary = lines[1].replace("Summary:", "").strip()
            line = '\n'.join(lines[2:])
            docs = [s.strip() for s in line.replace("Citations:", "").split(",")]
            themes.append(Theme(title=title, summary=summary, documents=docs))

    return ThemeResponse(themes=themes)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
