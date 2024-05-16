import os
from typing import List

from chromadb import Documents
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def from_documents(pages: List[Document]) -> Chroma:
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(documents=pages, embedding=embeddings, persist_directory='db')
    return db


def from_embeddings() -> Chroma:
    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory="db", embedding_function=embeddings)
    return db


def load_folder_data(db: Chroma, folder: str) -> Chroma:
    # Walk through the folder and its subfolders
    for dirpath, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            # Check if the file is not binary by trying to open it in text mode
            file_path = os.path.join(dirpath, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    # Save the content to the database along with the filename
                    doc = Document(page_content=content, metadata={"filename": filename, "file_path": file_path})
                    db.add_documents([doc])
            except UnicodeDecodeError:
                # Skip binary files that cannot be read in text mode
                print(f"Skipped binary file: {filename}")

    print("All files have been loaded into the database with filename information.")
    return db
