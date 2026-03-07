import openai
import os
from dotenv import load_dotenv

load_dotenv()

def test_raw_llm():
    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    )
    
    model = os.getenv("LLM_MODEL", "meta-llama/llama-3-8b-instruct")
    print(f"Testing model: {model}")
    
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say hello from SolarMind!"}],
        extra_headers={
            "HTTP-Referer": "https://solarmind.ai",
            "X-Title": "SolarMind AI"
        }
    )
    print(f"Response: {resp.choices[0].message.content}")

if __name__ == "__main__":
    test_raw_llm()
