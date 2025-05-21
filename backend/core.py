import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain import hub
from langchain_pinecone import PineconeVectorStore
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import Dict, Any

load_dotenv()

def run_llm(query: str, chat_history: list[Dict[str, Any]]):
    print("Running LLM... to answer: ", query)

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        deployment="text-embedding-3-small",
        model="text-embedding-3-small",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-12-01-preview",
    )

    llm_azure_openai = AzureChatOpenAI(
        temperature=0,
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        model=os.environ["AZURE_OPENAI_MODEL"],
    )

    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"],
        embedding=embeddings,
    )

    rag_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(
       llm=llm_azure_openai,
       retriever=vectorstore.as_retriever(),
       prompt=rephrase_prompt,
    )
    rag_chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=create_stuff_documents_chain(llm_azure_openai, rag_prompt),
    )

    result = rag_chain.invoke(input={"input": query, "chat_history": chat_history})

    new_result = {
        "query": query,
        "source_documents": result["context"],
        "result": result["answer"],
    }

    return new_result
