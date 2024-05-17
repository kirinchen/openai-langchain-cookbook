from langchain_openai import OpenAI

import main

if __name__ == '__main__':
    main.init_base()
    openai = OpenAI(model_name="code-davinci-002")
    text = "create a go helloword example"
    print(openai.invoke (text))
