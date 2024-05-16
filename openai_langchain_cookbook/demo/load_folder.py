import os

import main
from utils import asset_utils, chroma_utils
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

if __name__ == '__main__':
    main.init_base()
    path = os.path.join(os.path.dirname(main.__file__), 'assert', 'folder_data', 'java_project')
    db = chroma_utils.from_embeddings()
    chroma_utils.load_folder_data(db, path)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # 建構 QA Chain 來進行問答
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)
    query = "What Language in this project?"
    result = qa.invoke({"query": query})
    print(result)
    print(result["result"])
    query = "How to fix HelloWorld.main function?"
    result = qa.invoke({"query": query})
    print(result)
    print(result["result"])
