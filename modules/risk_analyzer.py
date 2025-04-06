import os
from modules.llm import call_llm
from modules.cache_utils import process_document_in_chunks

# -----------------------------
# Main Function
# -----------------------------

def analyze_contract_risks(rfp_text):
    """
    Analyzes contract-related risks from an RFP and suggests mitigation strategies.
    """
    # Process only first chunk of document for better performance
    rfp_chunk = rfp_text[:4000]
    
    prompt = f"""
You are an AI assistant tasked with analyzing contract risks in an RFP.

Step 1: Identify clauses that could pose risks to the bidder (e.g., unilateral termination rights, excessive penalties, unclear payment terms).
Step 2: Suggest modifications to balance the terms (e.g., adding a notice period for termination, capping penalties).

### RFP Text:

"""

    # Use centralized LLM call function
    result = call_llm(prompt)

    if "Error" in result:
        return result  # Return error message directly

    # Structured output
    structured_output = f"""
### ⚠️ RFP Risk Analysis and Mitigation Strategies

#### 🔍 1. Identified Risks
{result}

#### ✅ 2. Additional Considerations
- **Termination Rights**: Include a notice period or termination for cause only to protect vendors from abrupt contract termination.
- **Penalties**: Cap penalties or tie them to specific, measurable failures to ensure fairness.
- **Payment Terms**: Clearly define payment terms to ensure timely and predictable payments.

#### 📝 3. Summary of Recommendations
- **Pre-Proposal Conference**: Provide alternative access to materials for vendors who cannot attend.
- **Evaluation Criteria**: Define clear, objective criteria to ensure fairness.
- **Transparency**: Offer feedback to all bidders to build trust and improve future participation.
- **Termination Rights**: Add a notice period or termination for cause clause.
- **Penalties**: Cap penalties and tie them to measurable failures.
- **Payment Terms**: Ensure clear and timely payment terms.
"""
    return structured_output