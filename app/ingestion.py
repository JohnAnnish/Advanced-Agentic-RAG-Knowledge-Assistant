from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_DIR = Path("data/documents")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
)


def load_documents():
    chunks = []

    for pdf_path in DATA_DIR.glob("*.pdf"):
        pages = PyPDFLoader(str(pdf_path)).load()

        for chunk in text_splitter.split_documents(pages):
            chunk.metadata["source"] = pdf_path.name
            chunk.metadata["document_type"] = "pdf"
            chunks.append(chunk)

    return chunks
