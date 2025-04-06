from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import textwrap
from modules.llm import call_llm  # Use our centralized LLM module

def generate_summary_pdf(eligibility, checklist, risks, rfp_text, profile_text):
    """
    Generates a detailed, structured, and attractive executive summary PDF using LLM.
    """
    # Only use first part of documents to avoid token limits
    rfp_summary = rfp_text[:2000]
    profile_summary = profile_text[:2000]

    # Prompt to generate detailed, sectioned, and visually clear output
    prompt = f"""
You are an expert AI assistant helping companies analyze their readiness for RFPs. Based on the inputs below, generate a clear, 
professional, and visually structured executive report.

--- INPUTS ---

📄 RFP Summary:


🏢 Company Profile:
{profile_summary}

📋 Eligibility Report:
{eligibility}

✅ Submission Checklist:
{checklist}

⚠️ Contract Risk Analysis:
{risks}

--- OUTPUT FORMAT ---

Please structure the output as follows:

1. 📘 Executive Summary  
   - A 2–3 paragraph overview summarizing the opportunity and company fit.

2. ✅ Eligibility Matching  
   - Match eligibility criteria from the RFP with company capabilities.
   - Show a markdown-style in a list format:
     - List all mandatory criteria from the RFP.
     - List all matching items from the company profile.
     - Highlight any missing items or qualifications.
     - Provide a **final verdict**: Eligible or Not Eligible.
     - Include a **percentage match score**.
   - At the end, include a **percentage match score**.

3. 🔹 Submission Checklist Highlights  
   - 5–7 key points from the checklist, marked with ✅ or 🔸 for pending.

4. ⚠️ Risk Summary  
   - Explain potential risks (with ⚠️ markers).

5. 📈 Recommendations  
   - If eligibility is below 80%, provide action items to improve.
   - E.g., registration advice, missing documents, team expansion, etc.

Use markdown-like style (headers, bullets) for clarity.
"""

    # Call LLM with centralized function
    summary = call_llm(prompt)  # LLM-generated report (text)
    pdf_buffer = render_pdf(summary)  # Convert text to PDF
    return summary, pdf_buffer


def render_pdf(content: str):
    """
    Converts LLM-generated summary into a nicely formatted PDF using ReportLab.
    """
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setFont("Helvetica", 12)
    text_obj = pdf.beginText(40, 750)

    def flush_text(text_obj):
        pdf.drawText(text_obj)
        pdf.showPage()
        new_text = pdf.beginText(40, 750)
        new_text.setFont("Helvetica", 12)
        return new_text

    for line in content.split("\n"):
        # Keep markdown-style table formatting intact using monospace-like alignment
        if "|" in line and "-" not in line:
            wrapped = [line]  # Don't wrap table rows
        else:
            wrapped = textwrap.wrap(line, width=90, replace_whitespace=False)

        for sub_line in wrapped:
            text_obj.textLine(sub_line)
            if text_obj.getY() < 50:
                text_obj = flush_text(text_obj)

    pdf.drawText(text_obj)
    pdf.save()
    buffer.seek(0)
    return buffer