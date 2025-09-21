from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def export_report(query, reasoning, sources, answer, output_path="report.pdf"):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"📌 Query: {query}", styles["Heading2"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("💡 Reasoning:", styles["Heading3"]))
    for r in reasoning:
        story.append(Paragraph(r, styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("✅ Answer:", styles["Heading3"]))
    story.append(Paragraph(answer, styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("📚 Sources:", styles["Heading3"]))
    for i, src in enumerate(sources, start=1):
        story.append(Paragraph(f"{i}. {src.metadata.get('source', 'Unknown')} - {src.page_content[:800]}...", styles["Normal"]))
        story.append(Spacer(1, 12))

    doc.build(story)
