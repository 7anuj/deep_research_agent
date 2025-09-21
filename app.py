import streamlit as st
from io import BytesIO
import os

from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from report_generator import export_report
from retriever import load_hybrid_retriever
from reasoning_tracker import ReasoningTracker
# -------------------
# Setup
# -------------------
st.title("⚖️ Pocket Lawyer: Deep Researcher Agent")

# Memory for conversation
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"  # Only store answer in memory
)

# Initialize Ollama LLM
llm = Ollama(model="llama3", base_url="http://localhost:11434")

# Load prebuilt retriever
retriever = load_hybrid_retriever()

# Prompt with reasoning instructions
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a legal AI assistant. Use the given context from IPC/Constitution to answer the question.
Provide **step-by-step reasoning** for how you arrived at the answer.
If the answer is not in the context, clearly indicate it ("⚠️ No matching documents found").

Context:
{context}

Question:
{question}

Answer concisely, but accurately in legal terms, with reasoning steps.
"""
)

# Conversational chain with explicit output_key
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=True,
    output_key="answer"
)

# -------------------
# File Upload (user KB)
# -------------------
uploaded_file = st.file_uploader("Upload a legal PDF (optional)", type=["pdf"])
if uploaded_file:
    os.makedirs("data", exist_ok=True)
    filepath = f"data/{uploaded_file.name}"
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader(filepath)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        "storage", embeddings, allow_dangerous_deserialization=True)
    vectorstore.add_documents(docs)
    vectorstore.save_local("storage")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Replace retriever inside chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        output_key="answer"
    )
    st.success("✅ Uploaded PDF added to knowledge base!")

# -------------------
# Chat UI
# -------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show past messages
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

# User input
if query := st.chat_input("Ask your legal question..."):
    st.session_state.messages.append(("user", query))
    with st.chat_message("user"):
        st.markdown(query)

    # Run chain
    try:
        result = qa_chain({"question": query})
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
        result = {"answer": "⚠️ Could not generate an answer.",
                  "source_documents": []}

    # Fallback if no answer
    answer_text = result.get(
        "answer") or "⚠️ No matching documents found in knowledge base."

    # Step-by-step reasoning can be included in the answer itself from LLM
    st.session_state.messages.append(("assistant", answer_text))
    with st.chat_message("assistant"):
        st.markdown(answer_text)

        # Inline collapsible sources
        sources = result.get("source_documents", [])
        if sources:
            with st.expander("📚 Sources Used"):
                for i, doc in enumerate(sources, start=1):
                    snippet = doc.page_content[:500] + \
                        ("..." if len(doc.page_content) > 500 else "")
                    st.markdown(
                        f"**{i}. {doc.metadata.get('source', 'Unknown Source')}**\n\n{snippet}")

    # Save last interaction for export
    st.session_state.last_sources = sources
    st.session_state.last_query = query
    st.session_state.last_answer = answer_text

# -------------------
# Report Export
# -------------------
if st.button("📄 Export Report"):
    if "last_query" in st.session_state:
        # Export PDF to memory
        buffer = BytesIO()
        export_report(
            st.session_state.last_query,
            [m[1] for m in st.session_state.messages if m[0]
                == "user"],  # reasoning steps
            st.session_state.last_sources,
            st.session_state.last_answer,
            output_path=buffer  # export_report should handle file-like objects
        )
        buffer.seek(0)

        # Download button
        st.download_button(
            label="⬇️ Download Report (PDF)",
            data=buffer,
            file_name="legal_report.pdf",
            mime="application/pdf",
        )

        # Show sources inline in chat
        with st.chat_message("assistant"):
            st.markdown("### 📖 Sources Cited in Report")
            for i, doc in enumerate(st.session_state.last_sources, start=1):
                snippet = doc.page_content[:500] + \
                    ("..." if len(doc.page_content) > 500 else "")
                st.markdown(
                    f"**{i}. {doc.metadata.get('source', 'Unknown')}**\n\n{snippet}")

        st.success("✅ Report generated with sources!")
    else:
        st.warning("Ask a question first!")
