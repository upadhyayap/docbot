import os
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents import create_csv_agent
from langchain.agents import Tool
from typing import Any

load_dotenv()

def main():
    print("Starting the code interpreter...")

    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """
    
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    llm_azure_openai = AzureChatOpenAI(
        temperature=0,
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        model=os.environ["AZURE_OPENAI_MODEL"],
    )

    tools = [PythonREPLTool()]
    python_agent = create_react_agent(llm_azure_openai, tools, prompt)
    python_agent_executor = AgentExecutor(agent=python_agent, tools=tools, verbose=True)

    # input = """generate 2 qr codes and save it the current directory.
    # The code should point to 2 different langchain docs pages.
    # The qr codes should be named qr_code_1.png and qr_code_2.png.
    # you have qrcode library already installed.
    # """
    # agent_executor.invoke({"input": input})
    
    csv_agent_executor: AgentExecutor = create_csv_agent(
        llm=llm_azure_openai,
        path="data/data.csv",
        verbose=True,
        allow_dangerous_code=True
    )

    # input = "what is the average price of the products?"
    # csv_agent_executor.invoke({"input": input})

    ################################Agent routing###############################

    def python_agent_executor_wrapper(original_prompt: str) -> dict[str, Any]:
        return python_agent_executor.invoke({"input": original_prompt})

    tools = [
        Tool(
            name="Python Agent",
            func=python_agent_executor_wrapper,
            description="""useful when you need to transform natural language to python and execute the python code,
                          returning the results of the code execution
                          DOES NOT ACCEPT CODE AS INPUT""",
        ),
        Tool(
            name="CSV Agent",
            func=csv_agent_executor.invoke,
            description="""useful when you need to answer question over episode_info.csv file,
                         takes an input the entire question and returns the answer after running pandas calculations""",
        ),
    ]

    prompt = base_prompt.partial(instructions="")
    grand_agent = create_react_agent(
        prompt=prompt,
        llm=llm_azure_openai,
        tools=tools,
    )
    grand_agent_executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True)

    print(
        grand_agent_executor.invoke(
            {
                "input": "what is the average price of the products?",
            }
        )
    )

    print(
        grand_agent_executor.invoke(
            {
                "input": "Generate and save in current working directory 15 qrcodes that point to `www.udemy.com/course/langchain`",
            }
        )
    )
    


if __name__ == "__main__":
    main()
