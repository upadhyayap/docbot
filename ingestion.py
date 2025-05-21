import os
from dotenv import load_dotenv
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

def ingest_data():
    print("Ingesting data...")

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        deployment="text-embedding-3-small",
        model="text-embedding-3-small",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-12-01-preview",
    )

    dataDir = "langchain-docs/api.python.langchain.com/en/latest"
    loader = ReadTheDocsLoader(dataDir)
    raw_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50, separators=["\n"])
    documents = text_splitter.split_documents(raw_docs)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print("Indexing documents...")
    # Process in batches of 100 documents
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1} of {(len(documents) + batch_size - 1)//batch_size}")
        PineconeVectorStore.from_documents(batch, embeddings, index_name=os.environ["INDEX_NAME"])
        print(f"Batch {i//batch_size + 1} processed successfully")

    print("Data ingested successfully")

if __name__ == "__main__":
    ingest_data()

