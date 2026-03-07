import os
import structlog
import config
from typing import List, Optional

log = structlog.get_logger(__name__)

class LLMService:
    def __init__(self):
        self.openai_key = config.OPENAI_API_KEY
        self.gemini_key = config.GEMINI_API_KEY
        self.model = config.LLM_MODEL
        
        self.client = None
        self.provider = "none"

        if self.gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_key)
                self.client = genai.GenerativeModel(self.model)
                self.provider = "gemini"
                log.info("llm_provider_initialized", provider="gemini", model=self.model)
            except ImportError:
                log.warning("google_generativeai_not_installed", fallback="openai")
        
        if not self.client and self.openai_key:
            try:
                import openai
                client_kwargs = {"api_key": self.openai_key}
                if config.OPENAI_BASE_URL:
                    client_kwargs["base_url"] = config.OPENAI_BASE_URL
                self.client = openai.OpenAI(**client_kwargs)
                self.provider = "openai"
                log.info("llm_provider_initialized", provider="openai", model=self.model)
            except ImportError:
                log.warning("openai_not_installed")

    def generate_response(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        if not self.client:
            return "LLM unavailable or API key missing."
        
        try:
            if self.provider == "gemini":
                # Using Gemini API
                full_prompt = f"{system_prompt}\n\nUser Question: {prompt}"
                response = self.client.generate_content(full_prompt)
                return getattr(response, "text", str(response))
            elif self.provider == "openai":
                # Using OpenAI/OpenRouter API
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    extra_headers={
                        "HTTP-Referer": "https://solarmind.ai",
                        "X-Title": "SolarMind AI"
                    }
                )
                return resp.choices[0].message.content
            return "Unknown provider"
        except Exception as e:
            log.warning("llm_call_failed", provider=self.provider, error=str(e))
            return f"Error calling LLM ({self.provider}): {str(e)}"

# Global singleton
llm_service = LLMService()
