import os

from langchain_community.agent_toolkits.load_tools import load_tools

import main
from utils import asset_utils, chroma_utils
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType

if __name__ == '__main__':
    main.init_base()
    path = os.path.join(os.path.dirname(main.__file__), 'assert', 'folder_data', 'java_project')
    db = chroma_utils.from_embeddings()
    chroma_utils.load_folder_data(db, path)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    llm = OpenAI()
    tools = load_tools(
        ["llm-math"],
        llm=llm
    )

    agent = initialize_agent(
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        verbose=True,  # I will use verbose=True to check process of choosing tool by Agent
        max_iterations=3
    )
    # 建構 QA Chain 來進行問答
    qa = RetrievalQA.from_chain_type(
        llm=agent, chain_type="stuff", retriever=retriever, return_source_documents=True)
    query = "What Language in this project?"
    result = qa.invoke({"query": query})
    print(result)
    print(result["result"])
    query = "How to fix HelloWorld.main function?"
    result = qa.invoke({"query": query})
    print(result)
    print(result["result"])
