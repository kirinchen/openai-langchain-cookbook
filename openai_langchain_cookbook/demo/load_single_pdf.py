import os

import main
from utils import asset_utils, chroma_utils
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

if __name__ == '__main__':
    main.init_base()
    path = os.path.join(os.path.dirname(main.__file__), 'assert', 'ESG_LangChain.pdf')
    pages = asset_utils.gen_pdf_loader(path)
    print(pages)
    db = chroma_utils.from_documents(pages)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # 建構 QA Chain 來進行問答
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)
    query = "what is the main point in this book?"
    result = qa.invoke({"query": query})
    print(result)
    print(result["result"])
