# report_generator.py
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

llm = Ollama(model="llama3", base_url="http://localhost:11434")

summary_prompt = PromptTemplate(
    input_variables=["qa_pairs"],
    template="""
You are a research assistant. Summarize the following Q&A pairs into a structured, coherent research report.

Q&A Pairs:
{qa_pairs}

Format the report with:
1. Introduction
2. Key Findings
3. Detailed Analysis
4. Conclusion
"""
)

def generate_report(history):
    """Generate research report text from chat history"""
    qa_text = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in history])
    chain = summary_prompt | llm
    report_text = chain.invoke({"qa_pairs": qa_text})
    return report_text

def export_markdown(report_text, filename="research_report.md"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report_text)
    return filename

def export_pdf(report_text, filename="research_report.pdf"):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    story = []

    for line in report_text.split("\n"):
        if line.strip():
            story.append(Paragraph(line, styles["Normal"]))
            story.append(Spacer(1, 12))

    doc.build(story)
    return filename
