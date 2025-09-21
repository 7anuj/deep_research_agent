import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import EnsembleRetriever
from reasoning_tracker import ReasoningTracker
import tempfile

# Initialize LLM + Embeddings
llm = Ollama(model="llama3", base_url="http://localhost:11434")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")


# -------------------------------
# Prebuilt Retriever (FAISS index)
# -------------------------------
def load_prebuilt_retriever(storage_folder="storage"):
    """Load prebuilt FAISS index instead of rebuilding every time"""
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


def load_user_retriever(uploaded_files):
    """Create retriever for user-uploaded PDFs (Streamlit UploadedFile)"""
    docs = []

    for uploaded_file in uploaded_files:
        # Save UploadedFile to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs.extend(loader.load())

        # Optional: delete temp file after loading
        os.remove(tmp_path)

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})

# -------------------------------
# Combined Retriever + QA Chain
# -------------------------------


def build_combined_chain(prebuilt_retriever=None, user_retriever=None):
    """Merge retrievers if both exist, else fallback to whichever is available"""
    retrievers = []
    if prebuilt_retriever:
        retrievers.append(prebuilt_retriever)
    if user_retriever:
        retrievers.append(user_retriever)

    if len(retrievers) == 1:
        retriever = retrievers[0]
    else:
        retriever = EnsembleRetriever(
            retrievers=retrievers,
            weights=[0.5, 0.5]  # adjust if you want KB > user docs priority
        )

    tracker = ReasoningTracker()

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever,
        return_source_documents=True,
        callbacks=[tracker]
    )
    return qa_chain, tracker
