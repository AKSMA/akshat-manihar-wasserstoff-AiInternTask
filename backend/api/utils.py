from pathlib import Path
import shutil
from typing import List
from langchain_community.document_loaders import (
    UnstructuredPDFLoader, UnstructuredImageLoader,
    TextLoader, UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from config import UPLOAD_DIR, INDEX_DIR  # Import upload and index directories from config



def load_single_file(file_path: Path) -> List[Document]:
    """
    Loads a single file and returns a list of Document objects.
    Supports PDF, image, text, and Word document formats.
    Adds metadata for document ID and chunk index.
    """
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        # Load PDF using OCR
        loader = UnstructuredPDFLoader(str(file_path), mode="elements", strategy="ocr_only")
    elif ext in [".jpg", ".jpeg", ".png"]:
        # Load image files
        loader = UnstructuredImageLoader(str(file_path))
    elif ext == ".txt":
        # Load plain text files
        loader = TextLoader(str(file_path))
    elif ext == ".docx":
        # Load Word documents
        loader = UnstructuredWordDocumentLoader(str(file_path))
    else:
        # Raise error for unsupported file types
        raise ValueError(f"Unsupported file type: {ext}")

    docs = loader.load()
    # Add metadata to each document chunk
    for i, doc in enumerate(docs):
        doc.metadata["DocID"] = file_path.name
        doc.metadata["ChunkIndex"] = i
    return docs


def ingest_documents():
    """
    Loads all files from the upload directory, splits them into chunks,
    generates embeddings, and saves the FAISS vector index.
    Returns the number of original documents and the number of chunks.
    """
    all_docs = []
    # Iterate over all files in the upload directory
    for file_path in UPLOAD_DIR.glob("*"):
        try:
            docs = load_single_file(file_path)
            all_docs.extend(docs)
        except Exception as e:
            print(f"Failed to process {file_path.name}: {e}")

    # Split documents into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000, chunk_overlap=1500)
    chunked_docs = []
    for doc in all_docs:
        chunks = splitter.create_documents([doc.page_content], [doc.metadata])
        chunked_docs.extend(chunks)

    # Generate embeddings for each chunk using HuggingFace model
    embeddings=HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device':'cpu'},
        encode_kwargs={'normalize_embeddings':True}
    )
    # Create FAISS vector store and save locally
    vectordb = FAISS.from_documents(chunked_docs, embeddings)
    vectordb.save_local(INDEX_DIR)
    return len(all_docs), len(chunked_docs)