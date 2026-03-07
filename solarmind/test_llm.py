import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

import config
from rag.llm_service import llm_service

def test_llm():
    model = "meta-llama/llama-3-8b-instruct"
    print(f"Testing connectivity to OpenRouter with model: {model} and provider: {llm_service.provider}")
    print(f"Model: {llm_service.model}")
    print(f"Base URL: {config.OPENAI_BASE_URL}")
    
    prompt = "Reply with exactly 'System is UP' if you can read this."
    try:
        response = llm_service.generate_response(prompt)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_llm()
