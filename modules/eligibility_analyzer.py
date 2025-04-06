import os
import json
from modules.vector_store import VectorStore
from modules.llm import call_llm
from modules.cache_utils import process_document_in_chunks

# -----------------------------
# Basic Validation Functions
# -----------------------------

def is_valid_rfp(text):
    return "request for proposal" in text.lower() or "rfp" in text.lower()

def is_valid_company_profile(text):
    return any(keyword in text.lower() for keyword in ["about us", "experience", "services", "clients", "company"])

# -----------------------------
# Prompt Builder
# -----------------------------

def build_prompt(rfp_text, company_text, retrieved_context):
    # Use chunks of text instead of full documents
    rfp_chunk = rfp_text[:4000]
    company_chunk = company_text[:4000]
    
    return f"""
You are a compliance AI assistant helping a company respond to a government RFP.

First, check if the provided file is a valid RFP document and whether the company data document is relevant.
Then, proceed with the following analysis.

### Retrieved Context:
{retrieved_context}

### RFP Text:


### Company Profile:
{company_chunk}

Step 1: Extract all mandatory eligibility criteria (certifications, licenses, experience, etc.) from the RFP.
Step 2: Compare with the company profile and the retrieved context.
Step 3: Provide a clear bullet-point report:
- Mandatory Criteria Extracted
- Company Matches
- Missing Items (highlight any gaps in qualifications)
- Final Verdict: Eligible or Not Eligible
Step 4: Provide a summary of the analysis.
- Include notes or comments that may be relevant.
- Suggest actions the company can take to improve eligibility in future RFPs.
"""

# -----------------------------
# Result Parsing
# -----------------------------

def parse_analysis_sections(result):
    lines = result.strip().splitlines()
    mandatory, verdict, capture = [], "", False

    for line in lines:
        if "mandatory criteria" in line.lower():
            capture = "mandatory"
            continue
        if "final verdict" in line.lower():
            capture = "verdict"
            continue
        if capture == "mandatory" and line.strip().startswith("-"):
            mandatory.append(line)
        elif capture == "verdict" and line.strip():
            verdict = line.strip()
            break

    return "\n".join(mandatory) or "Could not extract mandatory criteria.", verdict or "Verdict not found."

# -----------------------------
# Main Orchestration Function
# -----------------------------

def run_eligibility_check(rfp_text, company_text):
    # Step 1: Validate documents
    if not is_valid_rfp(rfp_text):
        return "The uploaded RFP file does not appear to be a valid Request for Proposal document."

    if not is_valid_company_profile(company_text):
        return "The uploaded company profile does not seem to contain valid company data."

    # Step 2: Contextual Search
    vector_store = VectorStore()
    query = "What are the mandatory eligibility criteria for this RFP?"
    retrieved_docs = vector_store.search(query, top_k=3)

    retrieved_context = "\n".join([doc["text"] for doc in retrieved_docs]) if retrieved_docs else \
        "No relevant context found in the knowledge base."

    # Step 3: Build Prompt
    prompt = build_prompt(rfp_text, company_text, retrieved_context)

    # Step 4: Use centralized LLM call function
    result = call_llm(prompt)

    # Step 5: Parse and Structure Output
    if result.startswith("Error"):
        return result

    mandatory_criteria, verdict = parse_analysis_sections(result)

    structured_output = f"""
### 📝 Eligibility Analysis Report

#### ✅ 1. Mandatory Criteria Extracted
{mandatory_criteria}

#### 🏁 2. Final Verdict
{verdict}

#### 📊 3. Full Analysis Summary
{result}
"""
    return structured_output