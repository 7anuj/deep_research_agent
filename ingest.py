import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_prebuilt_index(data_folder="data", storage_folder="storage"):
    # Load all PDFs in prebuilt KB
    loader = DirectoryLoader(data_folder, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Save
    os.makedirs(storage_folder, exist_ok=True)
    vectorstore.save_local(storage_folder)
    print("✅ Prebuilt KB indexed successfully!")

if __name__ == "__main__":
    build_prebuilt_index()
