import os
import docx
import PyPDF2

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def extract_text(file):
    ext = os.path.splitext(file.name)[1].lower()
    if ext == ".docx":
        return extract_text_from_docx(file)
    elif ext == ".pdf":
        return extract_text_from_pdf(file)
    else:
        raise ValueError("Unsupported file type. Use PDF or DOCX.")
