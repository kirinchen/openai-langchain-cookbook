import openai
from langchain_openai import OpenAI, ChatOpenAI

import main

if __name__ == '__main__':
    main.init_base()
    openai = ChatOpenAI(model_name="gpt-4")
    text = "What would be a good company name for a company that makes colorful shoes?"
    print(openai.invoke (text))
