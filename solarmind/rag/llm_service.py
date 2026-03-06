import openai
import structlog
import config
from typing import List, Optional

log = structlog.get_logger(__name__)

class LLMService:
    def __init__(self):
        self.api_key = config.OPENAI_API_KEY
        self.base_url = config.OPENAI_BASE_URL
        self.model = config.LLM_MODEL
        
        if self.api_key:
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self.client = openai.OpenAI(**client_kwargs)
        else:
            self.client = None

    def generate_response(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        if not self.client:
            return "LLM unavailable or API key missing."
        
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            return resp.choices[0].message.content
        except Exception as e:
            log.warning("llm_call_failed", error=str(e))
            return f"Error calling LLM: {str(e)}"

# Global singleton
llm_service = LLMService()
