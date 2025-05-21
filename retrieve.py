from backend.core import run_llm

if __name__ == "__main__":
    query = "what are output parsers in langchain?"
    result = run_llm(query)
    print(result["result"])
