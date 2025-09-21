import os
import argparse
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def build_index(input_path, storage_folder="storage", mode="prebuilt"):
    """
    Build FAISS index from PDFs.
    mode = 'prebuilt'  -> entire folder of PDFs (knowledge base)
    mode = 'upload'    -> single user-uploaded PDF
    """

    if mode == "prebuilt":
        # Load all PDFs in knowledge base folder
        loader = DirectoryLoader(input_path, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()

    elif mode == "upload":
        # Load just one uploaded file
        loader = PyPDFLoader(input_path)
        documents = loader.load()

    else:
        raise ValueError("Mode must be 'prebuilt' or 'upload'")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load existing FAISS index if available
    if os.path.exists(storage_folder):
        vectorstore = FAISS.load_local(storage_folder, embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(docs)
    else:
        vectorstore = FAISS.from_documents(docs, embeddings)

    # Save
    os.makedirs(storage_folder, exist_ok=True)
    vectorstore.save_local(storage_folder)

    if mode == "prebuilt":
        print("✅ Prebuilt KB indexed successfully!")
    else:
        print(f"✅ Uploaded file '{input_path}' indexed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to folder (prebuilt) or file (upload)")
    parser.add_argument("--mode", type=str, choices=["prebuilt", "upload"], default="prebuilt")
    parser.add_argument("--storage", type=str, default="storage", help="Where to store FAISS index")
    args = parser.parse_args()

    build_index(args.input, args.storage, args.mode)
