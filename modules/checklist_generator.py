import os
from modules.llm import call_llm
from modules.cache_utils import process_document_in_chunks

# -----------------------------
# Main Function
# -----------------------------

def generate_submission_checklist(rfp_text):
    """
    Generates a submission checklist by analyzing the RFP text.
    It extracts document formatting, attachments, deadlines, and submission methods.
    """
    # Process only first chunk of document for better performance
    rfp_chunk = rfp_text[:4000]
    
    prompt = f"""
You are an AI assistant tasked with extracting submission requirements from an RFP.

Step 1: Identify all submission requirements, including:
- Document format (e.g., page limit, font type/size, line spacing).
- Specific attachments or forms required.
- Deadlines and submission methods.

Step 2: Provide a clear, structured checklist.

### RFP Text:

"""

    # Use centralized LLM call function
    result = call_llm(prompt)

    # Structure output nicely
    structured_output = f"""
### 📋 Submission Checklist

{result}
"""
    return structured_output