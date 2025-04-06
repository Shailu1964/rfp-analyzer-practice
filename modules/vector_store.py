import faiss
import os
from sentence_transformers import SentenceTransformer
import json

class VectorStore:
    def __init__(self, index_path="data/faiss_index"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # Hugging Face embedding model
        self.index_path = index_path
        self.index = None
        self.documents = []

        if os.path.exists(index_path):
            self.load_index()

    def load_index(self):
        """Load the FAISS index and associated documents."""
        self.index = faiss.read_index(self.index_path)
        with open(f"{self.index_path}_docs.json", "r") as f:
            self.documents = json.load(f)

    def save_index(self):
        """Save the FAISS index and associated documents."""
        faiss.write_index(self.index, self.index_path)
        with open(f"{self.index_path}_docs.json", "w") as f:
            json.dump(self.documents, f)

    def add_documents(self, docs):
        """Add documents to the vector store."""
        embeddings = self.model.encode(docs)
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
        self.index.add(embeddings)
        self.documents.extend(docs)
        self.save_index()

    def search(self, query, top_k=5):
        """Search for the most relevant documents."""
        if self.index is None:
            return []

        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)
        results = [{"text": self.documents[i], "distance": distances[0][j]} for j, i in enumerate(indices[0]) if i < len(self.documents)]
        return results