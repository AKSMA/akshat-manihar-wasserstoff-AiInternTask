from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
import os
import shutil

# LangChain and LLM imports
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import create_retrieval_chain

# Local imports
from models import QueryRequest, QueryResponse, ThemeResponse, DocAnswer, Theme
from utils import ingest_documents
from config import UPLOAD_DIR, INDEX_DIR
api_router = APIRouter(prefix="/api")

from config import groq_api_key

# Endpoint for uploading documents
@api_router.post("/upload/")
def upload_documents(files: List[UploadFile] = File(...)):
    # Limit the number of files to 75
    if len(files) > 75:
        return JSONResponse(content={"error": "Max 75 files allowed."}, status_code=400)
    # Save each uploaded file to the upload directory
    for file in files:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    # Ingest the uploaded documents for later retrieval
    doc_count, chunk_count = ingest_documents()
    return {"message": f"Uploaded {doc_count} documents.", "chunks": chunk_count}

# Endpoint to list all uploaded documents
@api_router.get("/documents/")
def list_documents():
    files = [{"filename": f.name} for f in UPLOAD_DIR.glob("*")]
    return {"documents": files}

# Endpoint to answer a query using the retrieval-augmented generation chain
@api_router.post("/query_chain", response_model=QueryResponse)
def query_chain(payload: QueryRequest):
    # Load embeddings and vector database
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever()

    # System prompt for the LLM
    system_prompt = """
    You are an AI assistant answering questions based ONLY on the context below.
    Each document snippet includes: DocID, Page, and ChunkIndex.

    - Use only the context.
    - Do not make up facts.
    - Cite using (DocID, Page, Chunk).
    """

    # Prompt template for the LLM
    prompt = ChatPromptTemplate.from_template('''
    Answer the following question only based on the context provided.
    Think step by step before providing a detailed answer.
    <context>
    {context}
    </context>                                        
    Question: {input}''')

    # Initialize the LLM
    llm = ChatGroq(groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",temperature=0)
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Run the retrieval and generation chain
    response = retrieval_chain.invoke({"input": payload.question})
    docs = response["context"]

    # Build citations list from retrieved documents
    citations = []
    for doc in docs:
        citations.append(DocAnswer(
            DocID=doc.metadata.get("DocID", "unknown"),
            Page=doc.metadata.get("page_number"),
            ChunkIndex=doc.metadata.get("ChunkIndex", -1),
            Excerpt=doc.page_content[:300].strip()
        ))

    return QueryResponse(answer=response["answer"], citations=citations)

# Endpoint to synthesize themes from relevant document chunks
@api_router.post("/synthesize", response_model=ThemeResponse)
def synthesize_themes(payload: QueryRequest):
    # Load embeddings and vector database
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever()
    # Retrieve relevant documents for the query
    docs = retriever.get_relevant_documents(payload.question)
    chunk_strs = []
    # Format each chunk for the LLM prompt
    for doc in docs:
        docid = doc.metadata.get("DocID", "unknown")
        page = doc.metadata.get("page_number", "-")
        chunk = doc.metadata.get("ChunkIndex", "-")
        excerpt = doc.page_content.strip().replace("\n", " ")[:400]
        chunk_strs.append(f"{docid} (Page {page}, Chunk {chunk}): {excerpt}")

    chunks_input = "\n".join(chunk_strs)

    # Prompt template for synthesizing themes
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

    # Initialize the LLM
    llm = ChatGroq(groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",temperature=0)
    final_prompt = prompt_template.format(
        question=payload.question, chunks=chunks_input)
    result = llm.invoke(final_prompt)
    themes = []
    # Parse the LLM output into Theme objects
    for block in result.content.split("Theme ")[1:]:
        lines = block.strip().split("\n")
        if len(lines) >= 3:
            title = lines[0].replace("–", "").strip()[2:]
            summary = lines[1].replace("Summary:", "").strip()
            line = '\n'.join(lines[2:])
            docs = [s.strip() for s in line.replace("Citations:", "").split(",")]
            themes.append(Theme(title=title, summary=summary, documents=docs))

    return ThemeResponse(themes=themes)