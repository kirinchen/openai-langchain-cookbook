import os

from langchain import hub
from langchain.agents import initialize_agent, AgentType, create_structured_chat_agent, AgentExecutor, \
    create_react_agent
from langchain.chains import RetrievalQA
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

import main
from utils import chroma_utils

if __name__ == '__main__':
    main.init_base()
    path = os.path.join(os.path.dirname(main.__file__), 'assert', 'folder_data', 'java_project')
    db = chroma_utils.from_embeddings()
    chroma_utils.load_folder_data(db, path)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    llm = ChatOpenAI(model_name="gpt-4")

    # Construct a QA Chain to handle queries

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever, return_source_documents=True,
                                        verbose=True, input_key="question")

    # tools = load_tools(
    #     ["llm-math"],
    #     llm=llm
    # )
    tools = [
        Tool(
            name="Project expert system",
            func=lambda query: chain.invoke({"question": query}),
            description="Relevant content or suggestions for modifications about the project.",
        ),
    ]

    # agent = initialize_agent(
    #     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #     tools=tools,
    #     llm=llm,
    #     verbose=True,  # Use verbose=True to check the process of choosing tool by Agent
    #     handle_parsing_errors=True  # Added to handle parsing errors
    # )
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, )
    query = "How to fix my project HelloWorld.main ? and show all fixed code."

    # query = "What Language in this project?"
    result = agent_executor.invoke({"input": query})
    print(result)
    # print(result["result"])
    # query = "How to fix my project HelloWorld.main ?"
    # result = agent.run(query)
    # print(result)
    # print(result["result"])
