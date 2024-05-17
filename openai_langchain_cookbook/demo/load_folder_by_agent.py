import os
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.tools import Tool

import main
from utils import asset_utils, chroma_utils
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI, ChatOpenAI
from langchain.agents import initialize_agent, AgentType

if __name__ == '__main__':
    main.init_base()
    path = os.path.join(os.path.dirname(main.__file__), 'assert', 'folder_data', 'java_project')
    db = chroma_utils.from_embeddings()
    chroma_utils.load_folder_data(db, path)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    llm = ChatOpenAI(model_name="gpt-4")

    # Construct a QA Chain to handle queries
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever, return_source_documents=True,
                                        verbose=True, input_key="question")

    # tools = load_tools(
    #     ["llm-math"],
    #     llm=llm
    # )
    tools = [
        Tool(
            name="Project expert system",
            func=lambda query: chain({"question": query}),
            description="Relevant content or suggestions for modifications about the project.",
        ),
    ]

    agent = initialize_agent(
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        verbose=True,  # Use verbose=True to check the process of choosing tool by Agent
        handle_parsing_errors=True  # Added to handle parsing errors
    )
    query = "How to fix my project HelloWorld.main ? and show all fixed code."

    # query = "What Language in this project?"
    result = agent.run(query)
    print(result)
    # print(result["result"])
    # query = "How to fix my project HelloWorld.main ?"
    # result = agent.run(query)
    # print(result)
    # print(result["result"])
