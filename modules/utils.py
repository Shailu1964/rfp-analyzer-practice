from modules.llm import call_llm

def ask_query_from_context(query, rfp_text, profile_text, eligibility, checklist, risks):
    prompt = f"""
You are an intelligent assistant helping a company understand and respond to an RFP.

Below is all the relevant information:

=== RFP Summary ===
{rfp_text[:3000]}

=== Company Profile ===
{profile_text[:2000]}

=== Eligibility Report ===
{eligibility}

=== Submission Checklist ===
{checklist}

=== Contract Risk Analysis ===
{risks}

Based on the above content, answer this user query:
"{query}"

Respond clearly and helpfully. If something is missing or unclear, say so.
    """
    return call_llm(prompt)
