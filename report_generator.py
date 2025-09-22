# report_generator.py
from io import BytesIO
import io
import markdown2
from fpdf import FPDF
import os

# ------------------------
# Generate Report Text
# ------------------------


def generate_report(history):
    report = ""
    for i, chat in enumerate(history, 1):
        report += f"Q{i}: {chat['question']}\n"
        report += f"A{i}: {chat['answer']}\n"
        report += "Sources:\n"
        for doc in chat["sources"]:
            report += f"- {doc.metadata.get('source', 'Unknown')}\n"
        report += "\n"
    return report

# ------------------------
# Export as Markdown
# ------------------------
def export_markdown(report_text):
    """
    Convert report text to Markdown bytes
    """
    md_bytes = report_text.encode("utf-8")
    return md_bytes


# ------------------------
# Export as PDF
# ------------------------
# report_generator.py

def export_pdf(text):
    """
    Export text report to a PDF with Unicode support using DejaVuSans.
    Returns a BytesIO object for Streamlit download.
    """
    pdf = FPDF()
    pdf.add_page()

    # Path to DejaVu font folder
    font_folder = os.path.join(os.path.dirname(__file__), "C:\work\deep_research_agent\dejavusans")
    ttf_path = os.path.join(font_folder, "DejaVuSans.ttf")

    # Add DejaVu font (Unicode)
    pdf.add_font("DejaVu", "", ttf_path, uni=True)
    pdf.set_font("DejaVu", "", 12)

    # Replace bullets and write lines
    for line in text.split("\n"):
        line = line.replace("* ", "• ")
        pdf.multi_cell(0, 8, line)

    # Export to bytes
    pdf_bytes = pdf.output(dest='S').encode('latin1', 'replace')  # returns str, encode to bytes
    return io.BytesIO(pdf_bytes)
