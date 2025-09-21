import streamlit as st
import tempfile
from retriever_manager import (
    load_prebuilt_retriever,
    load_user_retriever,
    build_combined_chain,
)
from report_generator import generate_report, export_markdown, export_pdf
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain



# --------------------------
# Sidebar: Mode Selection
# --------------------------
st.sidebar.title("⚙️ Options")
mode = st.sidebar.radio(
    "Choose Mode:",
    ["Research Knowledge Base", "Upload Your Docs", "Hybrid (KB + User Docs)"]
)

# --------------------------
# Session State Initialization
# --------------------------
if "history_detailed" not in st.session_state:
    st.session_state.history_detailed = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
    st.session_state.tracker = None
if "prebuilt_retriever" not in st.session_state:
    st.session_state.prebuilt_retriever = None

# --------------------------
# LLM Summarizer (Strict RAG)
# --------------------------
llm_summarizer = Ollama(model="llama3", base_url="http://localhost:11434")
SUMMARY_PROMPT = PromptTemplate(
    input_variables=["raw_text"],
    template="""
You are a friendly, professional, and precise legal research assistant. Your answers must be based **only** on the retrieved documents. 

Instructions:

1. Read the raw answer carefully and summarize it into a natural, clear paragraph.
2. Highlight any legal references, such as **IPC sections**, **Constitution articles**, or other statutes mentioned in the documents.
3. Include bullet points for key points if they make the explanation clearer.
4. Do **not** add any information that is not present in the documents.
5. If the documents do not provide relevant information, clearly state: "No answer found based on the available context."

Raw text:
{raw_text}

Return a natural, professional, well-structured paragraph that includes all references, followed by bullet points if necessary.
"""
)


def summarize_answer(raw_text):
    llm_chain = LLMChain(llm=llm_summarizer, prompt=SUMMARY_PROMPT)
    return llm_chain.run(raw_text).strip()

# --------------------------
# Helper: Process Uploaded PDFs
# --------------------------


def process_uploaded_files(uploaded_files):
    temp_paths = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            temp_paths.append(tmp.name)
    return temp_paths

# --------------------------
# Helper: Build User QA Chain
# --------------------------


@st.cache_resource(show_spinner=False)
def build_user_chain(user_paths):
    user_retriever = load_user_retriever(user_paths)
    qa_chain, tracker = build_combined_chain(user_retriever=user_retriever)
    return qa_chain, tracker


# --------------------------
# Main App Title
# --------------------------
st.title("🤖 Legal Research Assistant")

# --------------------------
# Build QA Chain based on Mode
# --------------------------
if mode == "Research Knowledge Base":
    st.header("📚 Research Knowledge Base Mode")
    if st.session_state.prebuilt_retriever is None:
        st.session_state.prebuilt_retriever = load_prebuilt_retriever()
    st.session_state.qa_chain, st.session_state.tracker = build_combined_chain(
        prebuilt_retriever=st.session_state.prebuilt_retriever
    )

elif mode == "Upload Your Docs":
    st.header("📂 User Upload Mode")
    # We will handle the upload at the bottom only

elif mode == "Hybrid (KB + User Docs)":
    st.header("🔀 Hybrid Mode")
    if st.session_state.prebuilt_retriever is None:
        st.session_state.prebuilt_retriever = load_prebuilt_retriever()
    # Upload handled at bottom

# --------------------------
# Chat Section
# --------------------------
# Display past messages
for chat in st.session_state.history_detailed:
    with st.chat_message("user"):
        st.markdown(chat["question"])
    with st.chat_message("assistant"):
        st.markdown(chat["answer"])
        if chat["sources"]:
            with st.expander("📚 Sources Used"):
                for i, doc in enumerate(chat["sources"], 1):
                    snippet = doc.page_content[:500] + \
                        ("..." if len(doc.page_content) > 500 else "")
                    st.markdown(
                        f"**{i}. {doc.metadata.get('source', 'Unknown')}**\n\n{snippet}")

# User input
if query := st.chat_input("Ask your legal question..."):
    with st.chat_message("user"):
        st.markdown(query)

    try:
        result = st.session_state.qa_chain({
            "question": query,
            "chat_history": [(h["question"], h["answer"]) for h in st.session_state.history_detailed]
        })
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
        result = {"answer": "⚠️ Could not generate an answer.",
                  "source_documents": []}

    # Summarize strictly using RAG
    summarized_answer = summarize_answer(result["answer"])

    with st.chat_message("assistant"):
        st.markdown(summarized_answer)
        sources = result.get("source_documents", [])
        if sources:
            with st.expander("📚 Sources Used"):
                for i, doc in enumerate(sources, 1):
                    snippet = doc.page_content[:500] + \
                        ("..." if len(doc.page_content) > 500 else "")
                    st.markdown(
                        f"**{i}. {doc.metadata.get('source', 'Unknown')}**\n\n{snippet}")

    # Update history
    st.session_state.history_detailed.append({
        "question": query,
        "answer": summarized_answer,
        "sources": sources
    })

# --------------------------
# Bottom PDF Uploader (Optional)
# --------------------------
with st.expander("📂 Upload PDFs for context (optional)"):
    uploaded_files = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True, key="bottom_uploader"
    )
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        user_paths = process_uploaded_files(uploaded_files)
        st.session_state.qa_chain, st.session_state.tracker = build_user_chain(
            user_paths)
        st.success("📄 Uploaded documents are now available for retrieval.")

# --------------------------
# Export Section
# --------------------------
if st.session_state.history_detailed:
    st.subheader("📑 Generate Research Report")
    if st.button("Summarize Session"):
        report_text = generate_report(st.session_state.history_detailed)
        st.session_state.report = report_text
        st.write("### 📝 Research Report")
        st.write(report_text)

    if "report" in st.session_state:
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Download as Markdown",
                export_markdown(st.session_state.report),
                file_name="research_report.md"
            )
        with col2:
            st.download_button(
                "⬇️ Download as PDF",
                export_pdf(st.session_state.report),
                file_name="research_report.pdf"
            )
