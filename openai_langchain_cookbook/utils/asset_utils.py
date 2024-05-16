from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def gen_pdf_loader(file_path: str) -> List[Document]:
    py_mu_pdf_loader = PyMuPDFLoader(file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    py_mu_pdf_doc = py_mu_pdf_loader.load_and_split(splitter)
    return py_mu_pdf_doc
