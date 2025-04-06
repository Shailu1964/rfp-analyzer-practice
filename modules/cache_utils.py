import hashlib
import os
import pickle
import time
import random

def cached_llm_call(prompt, backend="groq", model=None, cache_dir="cache"):
    """
    Wrapper for LLM calls with disk caching to avoid redundant API calls.
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a cache key from the prompt, backend and model
    cache_key = hashlib.md5(f"{prompt}:{backend}:{model}".encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    # Check if we have a cached response
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            # If loading cache fails, continue to generate a new response
            pass
    
    # No cache or cache failed, make the actual LLM call
    if backend == "groq":
        from modules.llm import run_groq_analysis_with_backoff
        response = run_groq_analysis_with_backoff(prompt, model)
    else:
        from modules.llm import run_ollama_analysis
        response = run_ollama_analysis(prompt, model)
    
    # Cache the response
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(response, f)
    except Exception as e:
        print(f"Failed to cache response: {e}")
    
    return response

def process_document_in_chunks(text, chunk_size=2000, overlap=100):
    """Process a document in overlapping chunks to handle large documents."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # If this isn't the first chunk, include overlap from previous chunk
        if start > 0:
            start = max(0, start - overlap)
        
        chunks.append(text[start:end])
        start = end
        
    return chunks