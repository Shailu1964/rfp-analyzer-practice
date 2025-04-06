import os
import requests
import time
import random
import ollama

def run_ollama_analysis(prompt, model="llama2"):
    """
    Sends prompt to a local LLM model via Ollama.
    """
    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return response['message']['content']
    except Exception as e:
        return f"Error running Ollama analysis: {e}"

def run_groq_analysis_with_backoff(prompt, model=None, max_retries=5):
    """
    Sends prompt to Groq's API with exponential backoff for rate limits.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY is not set in the environment variables."

    model = model or os.getenv("GROQ_MODEL", "llama3-8b-8192")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    json_data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[Groq] Attempt {attempt}/{max_retries}...")
            res = requests.post("https://api.groq.com/openai/v1/chat/completions", 
                               headers=headers, json=json_data)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                # Exponential backoff with jitter
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"[Groq] Rate limited, retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                continue
            print(f"[Groq] Attempt {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                break
            time.sleep(1)  # Brief pause before next attempt
        except requests.exceptions.RequestException as e:
            print(f"[Groq] Attempt {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                break
            time.sleep(1)  # Brief pause before next attempt
    
    # If we get here, all retries failed
    print("[Groq] All retries failed, falling back to Ollama...")
    return run_ollama_analysis(prompt)

def call_llm(prompt, backend=None, model=None):
    """
    Central function to call LLM, determining backend from environment if not specified.
    """
    from modules.cache_utils import cached_llm_call
    
    backend = backend or os.getenv("LLM_BACKEND", "ollama").lower()
    return cached_llm_call(prompt, backend, model)