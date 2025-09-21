import streamlit as st
from retriever_manager import load_prebuilt_retriever, load_user_retriever, build_combined_chain
from report_generator import generate_report, export_markdown, export_pdf

st.sidebar.title("⚙️ Options")
mode = st.sidebar.radio(
    "Choose Mode:",
    ["Research Knowledge Base", "Upload Your Docs", "Hybrid (KB + User Docs)"]
)

qa_chain, tracker = None, None

if mode == "Research Knowledge Base":
    st.header("🤖 Deep Researcher (Knowledge Base Mode)")
    prebuilt = load_prebuilt_retriever()
    qa_chain, tracker = build_combined_chain(prebuilt_retriever=prebuilt)

elif mode == "Upload Your Docs":
    st.header("📂 Deep Researcher (User Upload Mode)")
    uploaded_files = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        user_retriever = load_user_retriever(uploaded_files)
        qa_chain, tracker = build_combined_chain(user_retriever=user_retriever)

elif mode == "Hybrid (KB + User Docs)":
    st.header("🔀 Deep Researcher (Hybrid Mode)")
    uploaded_files = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True)
    prebuilt = load_prebuilt_retriever()
    if uploaded_files:
        user_retriever = load_user_retriever(uploaded_files)
        qa_chain, tracker = build_combined_chain(prebuilt, user_retriever)
    else:
        qa_chain, tracker = build_combined_chain(prebuilt)

# --- Chat ---
if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask your research question:")

if query and qa_chain:
    tracker.steps = []
    result = qa_chain(
        {"question": query, "chat_history": st.session_state.history})
    st.session_state.history.append((query, result["answer"]))

    st.write("### 📝 Answer")
    st.write(result["answer"])

    st.write("### 📚 Sources")
    for doc in result["source_documents"]:
        st.write(f"- {doc.metadata.get('source', 'Unknown')}")

    st.write("### 🔎 Reasoning Trace")
    for step in tracker.get_steps():
        st.write(f"- {step}")

# --- Summarization + Export ---
if st.session_state.history:
    st.subheader("📑 Generate Research Report")

    if st.button("Summarize Session"):
        report_text = generate_report(st.session_state.history)
        st.session_state.report = report_text
        st.write("### 📝 Research Report")
        st.write(report_text)

    if "report" in st.session_state:
        col1, col2 = st.columns(2)

        with col1:
            if st.download_button(
                "⬇️ Download as Markdown",
                export_markdown(st.session_state.report),
                file_name="research_report.md"
            ):
                st.success("Markdown report ready!")

        with col2:
            if st.download_button(
                "⬇️ Download as PDF",
                export_pdf(st.session_state.report),
                file_name="research_report.pdf"
            ):
                st.success("PDF report ready!")
