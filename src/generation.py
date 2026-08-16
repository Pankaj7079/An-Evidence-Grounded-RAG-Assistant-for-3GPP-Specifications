"""Evidence-Grounded Generation Engine with Dynamic Technical Structuring.

Supports high-throughput Groq models (Llama-3.1-8b, Llama-3-70b, Qwen-2.5, DeepSeek)
and Google Gemini with automatic rate-limit resilience.
"""

import logging
import re
import time
from typing import List, Optional

from config import settings
from src.models import RetrievalResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Principal 3GPP Telecom System Architect assisting engineering teams with 3GPP 5G Specifications (TS 23.501 Architecture & TS 23.502 Procedures).

PRIMARY DIRECTIVE:
You must answer questions strictly and exclusively using the provided 3GPP specification excerpts.

ANSWER STRUCTURING & CITATION RULES:
1. Provide a professional, authoritative engineering response with dynamically chosen, topic-appropriate Markdown subheadings:
   - For architectural functions (e.g. AMF, UPF, SMF):
     `### 1. Architectural Overview [TS 23.501 Clause X.Y, Page Z]`
     `### 2. Core Functional Responsibilities`
     `### 3. Supported Interfaces & Reference Points`
   - For procedures & flows (e.g. Registration, PDU Session Establishment, Handover):
     `### 1. Procedure Purpose & Triggers [TS 23.502 Clause X.Y, Page Z]`
     `### 2. Step-by-Step Signaling & Execution Flow`
     `### 3. Network Functions & Reference Points Involved`
   - For specific security or feature mechanisms (e.g. IPUPS, Network Slicing):
     Use headings that directly reflect the feature's technical operations (e.g. `### 1. Border Security & Deployment Architecture`, `### 2. Packet Filtering & Tunnel Termination`, `### 3. Control Plane Interaction via N4`).

2. ZERO-REPETITION CITATION RULE:
   - Include the exact specification citation in the section heading or initial summary: `[TS 23.501 Clause X.Y, Page Z]` or `[TS 23.502 Clause X.Y, Page Z]`.
   - DO NOT repeat the same citation on every single sub-bullet line. Group related technical points under the appropriate section and cite once per section/clause.
   - Use bold descriptive lead-ins for each bullet point based on its actual technical meaning.

3. STRICT NEGATIVE QUERY & ABSTENTION RULE:
   - If the user query asks about a concept, procedure, or feature that does NOT exist in the provided 3GPP context (e.g. 6G, quantum teleportation, non-telecom questions), output ONLY this single concise sentence:
     "I could not find sufficient supporting evidence in the indexed 3GPP documents (TS 23.501 & TS 23.502) for this query."
   - DO NOT generate section headers, bullet lists, or dummy citations for negative queries.
   - NEVER include robotic meta-commentary like "There is no mention of TS 23.502 in the provided context...", "Based on the provided excerpts...", or "According to the retrieved text...".
"""


def clean_output_formatting(text: str) -> str:
    """Clean up formatting, empty backticks, and ensure clean presentation."""
    # If the text is an abstention, ensure it has no extraneous section headers
    if "could not find sufficient supporting evidence" in text.lower():
        for line in text.split("\n"):
            if "could not find sufficient supporting evidence" in line.lower():
                return line.strip()

    # Remove empty backticks or double spaces
    text = re.sub(r"\s*``\s*", " ", text)
    text = re.sub(r"\s*''\s*", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


class LLMClient:
    """Unified LLM client interface with multi-model fallback."""

    def __init__(
        self,
        provider: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        groq_model: Optional[str] = None,
        gemini_model: Optional[str] = None,
    ):
        self.provider = provider or settings.LLM_PROVIDER
        self.groq_api_key = groq_api_key or settings.GROQ_API_KEY
        self.gemini_api_key = gemini_api_key or settings.GEMINI_API_KEY
        self.groq_model = groq_model or settings.GROQ_MODEL
        self.gemini_model = gemini_model or "gemini-flash-latest"

        # Modern active Groq models in priority order
        self.groq_models = [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "llama3-70b-8192",
            "llama3-8b-8192",
        ]

        # Initialize provider client
        self._groq_client = None
        self._gemini_client = None

        if self.groq_api_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=self.groq_api_key, max_retries=0)
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")

        if self.gemini_api_key:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=self.gemini_api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}")

    def generate(
        self,
        prompt: str,
        system_instruction: str = SYSTEM_PROMPT,
        temperature: float = 0.0,
        max_tokens: int = settings.MAX_OUTPUT_TOKENS,
    ) -> str:
        """Generate deterministic response with seamless multi-model fallback."""
        # 1. Try Groq models in sequence
        if self._groq_client:
            for model_name in self.groq_models:
                try:
                    res = self._generate_groq(prompt, system_instruction, temperature, max_tokens, model=model_name)
                    return clean_output_formatting(res)
                except Exception as e:
                    err_msg = str(e).lower()
                    if "429" in err_msg or "rate limit" in err_msg:
                        logger.warning(f"Groq {model_name} rate limit, brief pause...")
                        time.sleep(2.0)
                    else:
                        logger.warning(f"Groq {model_name} failed: {e}")

        # 2. Try Gemini
        if self._gemini_client:
            try:
                res = self._generate_gemini(prompt, system_instruction, temperature, max_tokens)
                return clean_output_formatting(res)
            except Exception as e:
                logger.warning(f"Gemini generation failed: {e}.")

        # 3. Final single attempt on primary groq model after backoff
        if self._groq_client:
            time.sleep(3.0)
            res = self._generate_groq(prompt, system_instruction, temperature, max_tokens, model="llama-3.1-8b-instant")
            return clean_output_formatting(res)

        raise RuntimeError("All LLM providers and fallback models are currently unavailable.")

    def _generate_groq(
        self,
        prompt: str,
        system_instruction: str,
        temperature: float,
        max_tokens: int,
        model: Optional[str] = None,
    ) -> str:
        target_model = model or "llama-3.1-8b-instant"
        response = self._groq_client.chat.completions.create(
            model=target_model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def _generate_gemini(
        self, prompt: str, system_instruction: str, temperature: float, max_tokens: int
    ) -> str:
        full_content = f"{system_instruction}\n\nUser Question:\n{prompt}"
        response = self._gemini_client.models.generate_content(
            model=self.gemini_model,
            contents=full_content,
        )
        return response.text.strip()


def format_grounded_prompt(
    query: str,
    retrieved_chunks: List[RetrievalResult],
) -> str:
    """Format prompt with compact 3GPP clause excerpts to stay within rate limits."""
    context_blocks = ["=== 3GPP SPECIFICATION CONTEXT EXCERPTS ==="]
    for idx, chunk in enumerate(retrieved_chunks, 1):
        # Truncate chunk text to ~800 chars to maintain compact prompt tokens
        clean_chunk_text = chunk.text.strip()
        if len(clean_chunk_text) > 850:
            clean_chunk_text = clean_chunk_text[:850] + "..."

        context_blocks.append(
            f"--- Context Source [{idx}] ---\n"
            f"Specification: {chunk.document_code} (Rel-{chunk.release})\n"
            f"Clause: {chunk.section_number} - {chunk.section_title}\n"
            f"Page Number: {chunk.page_number}\n"
            f"Content:\n{clean_chunk_text}\n"
        )

    context_str = "\n".join(context_blocks)
    user_prompt = (
        f"{context_str}\n\n"
        f"=== USER QUESTION ===\n{query}\n\n"
        f"Provide an evidence-grounded technical answer based strictly on the context above. "
        f"Include exact citations in section headings (e.g. `[TS 23.501 Clause X.Y, Page Z]`). Do not repeat the same citation inside every bullet point."
    )
    return user_prompt
