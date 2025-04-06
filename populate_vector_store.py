from modules.vector_store import VectorStore

documents = [
    "ISO 9001 certification is required for eligibility.",
    "The bidder must have a Federal Contractor License.",
    "The project requires 5+ years of experience in government contracts.",
    "Unilateral termination clauses should be avoided in contracts."
]

vector_store = VectorStore()
vector_store.add_documents(documents)
print("Documents added to the vector store.")