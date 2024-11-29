from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import BedrockEmbeddings
import boto3

# Set up the Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")  # Update your region
bedrock_embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v1', client=bedrock)

def data_ingestion():
    # Load PDF documents from the './data' directory
    loader = PyPDFDirectoryLoader("./data")
    documents = loader.load()

    # Split documents into chunks of 1000 characters with a 100-character overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    return docs

def get_vector_store(docs):
    # Create FAISS vector store from documents and embeddings
    vector_store_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    # Save the FAISS index locally
    vector_store_faiss.save_local("faiss_index")

if __name__ == '__main__':
    # Ingest data and generate vector store
    docs = data_ingestion()
    get_vector_store(docs)