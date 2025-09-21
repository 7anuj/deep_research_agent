import os
import tempfile
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import EnsembleRetriever
from reasoning_tracker import ReasoningTracker

# -------------------------------
# Initialize LLM + Embeddings
# -------------------------------
llm = Ollama(model="llama3", base_url="http://localhost:11434")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------
# Prebuilt Retriever (FAISS index)
# -------------------------------
def load_prebuilt_retriever(storage_folder="storage"):
    """
    Load prebuilt FAISS index instead of rebuilding every time.
    Returns a retriever.
    """
    if os.path.exists(storage_folder):
        vectorstore = FAISS.load_local(
            storage_folder,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vectorstore.as_retriever(search_kwargs={"k": 5})
    else:
        raise FileNotFoundError(
            f"❌ No FAISS index found at {storage_folder}. Run ingest.py first to build it."
        )

# -------------------------------
# User Retriever (on-the-fly)
# -------------------------------
def load_user_retriever(file_paths):
    """
    Create a retriever for user-uploaded PDFs given as file paths.
    This builds a temporary FAISS vector store only for user docs.
    """
    if not file_paths:
        return None

    docs = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())

    if not docs:
        return None

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})

# -------------------------------
# Combined Retriever + QA Chain
# -------------------------------
def build_combined_chain(prebuilt_retriever=None, user_retriever=None):
    """
    Merge prebuilt and user retrievers if both exist.
    Falls back to whichever is available.
    Returns (qa_chain, tracker)
    """
    retrievers = []
    if prebuilt_retriever:
        retrievers.append(prebuilt_retriever)
    if user_retriever:
        retrievers.append(user_retriever)

    if not retrievers:
        raise ValueError("At least one retriever must be provided!")

    if len(retrievers) == 1:
        combined_retriever = retrievers[0]
    else:
        combined_retriever = EnsembleRetriever(
            retrievers=retrievers,
            weights=[0.5] * len(retrievers)  # equal weighting; adjust if needed
        )

    tracker = ReasoningTracker()

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        combined_retriever,
        return_source_documents=True,
        callbacks=[tracker]
    )

    return qa_chain, tracker
