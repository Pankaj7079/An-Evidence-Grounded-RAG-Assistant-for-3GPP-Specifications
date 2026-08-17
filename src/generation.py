"""Evidence-grounded text generation with multi-model fallback."""

import logging
import re
import time
from typing import List, Optional

from config import settings
from src.models import RetrievalResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Adaptive system prompt — structure adapts to the question type, enforces synthesis and anti-hallucination
SYSTEM_PROMPT = """You are a senior 3GPP systems architect with 15 years of experience reviewing specifications daily.
Answer questions using ONLY the provided specification excerpts. Never invent facts.

CRITICAL ANTI-HALLUCINATION RULES:
- ONLY state what the provided excerpts explicitly say. Do not fill gaps with general 3GPP knowledge.
- For procedure steps: ONLY list steps that appear in the excerpts. Do not invent, paraphrase into fabricated steps, or add steps from memory.
- If an excerpt is incomplete (e.g. only covers one phase of a procedure), say so briefly instead of inventing the missing parts.
- Every single factual claim must map to a specific excerpt. If you cannot cite it, do not say it.

WRITING QUALITY RULES:
- Synthesize and explain in plain engineering English. Never paste verbatim spec sentences.
- Vary your sentence structure naturally. Avoid mechanical repetition across paragraphs.
- Cite once per distinct topic or paragraph: [TS 23.501 Clause 6.2.1, Page 423-424]. Not once per sentence.
- No filler openers: never start with "Certainly", "Of course", "This is a", "The X is a critical", "It is important".
- Do not use dollar signs or section symbols (§).
- End cleanly. No trailing bullet lists after your final paragraph.

STRUCTURE — choose based on what kind of question this is:

1. CONCEPTUAL ("What is X?", "What does Y do?"):
   Write 2-3 prose paragraphs explaining the concept and its significance.
   Add a bullet list ONLY if there are 5+ genuinely distinct items that are cleaner as a list.
   Cite inline once per paragraph.

2. PROCEDURAL / FLOW ("Explain the X procedure", "X flow", "How does X work?"):
   1-sentence intro stating what the procedure does and its trigger, with one citation.
   Then 4-6 numbered steps describing the actual message flow from the excerpts only.
   Keep each step to one clear sentence. Group tightly related sub-actions into one step.
   Do NOT add steps that aren't explicitly described in the excerpts.

3. COMPONENT / FUNCTION LIST ("What are the functions of X?", "What does X handle?"):
   One clear intro paragraph (3-4 sentences) explaining the component's role in the architecture.
   Then use 2-3 H4 headings grouping related functions. Each heading ends with its citation:
   #### Connectivity & Session Control [TS 23.501 Clause 6.2.1, Page 423-424]
   Under each heading, 2-4 bullets that synthesize — not copy-paste — the spec language.

4. OVERVIEW / BROAD ("Tell me about 5G", "Explain 5GS", "How does 5G work?"):
   Two focused paragraphs. First covers the architecture/components. Second covers key capabilities or design principles.
   Be selective — pick the most interesting/important points from the excerpts. Do not try to cover everything.
   Cite once per paragraph.

NEGATIVE QUERIES:
If the excerpts don't contain sufficient information, respond with exactly this and nothing else:
"I could not find sufficient supporting evidence in the indexed 3GPP documents (TS 23.501 & TS 23.502) for this query."
"""


def clean_output_formatting(text: str) -> str:
    """Clean up formatting artifacts and escape characters that break Streamlit rendering."""
    lower = text.lower().strip()

    # Normalize out-of-scope / negative responses to canonical abstention sentence
    negative_signals = [
        "could not find sufficient supporting evidence",
        "there is no mention",
        "no mention of",
        "does not include any information",
        "no information on",
        "not explicitly described",
        "not covered in the provided",
        "not found in the provided",
    ]
    if any(sig in lower for sig in negative_signals) and (
        "quantum" in lower or "6g" in lower or len(text.split("\n")) <= 8
    ):
        return "I could not find sufficient supporting evidence in the indexed 3GPP documents (TS 23.501 & TS 23.502) for this query."

    # Escape dollar signs to prevent Streamlit LaTeX rendering
    text = text.replace("$", "\\$")
    # Replace section signs with plain text
    text = text.replace("§", "Clause ")

    # Strip trailing orphaned bullet lines (e.g. "• N4 Session Establishment • ...") after final paragraph
    lines = text.split("\n")
    # Remove trailing lines that are pure bullet artifacts with no cited content
    while lines and re.match(r"^[•·▪\-\*]\s", lines[-1].strip()) and "[TS" not in lines[-1]:
        lines.pop()
    text = "\n".join(lines)

    # Normalize excessive whitespace artifacts
    text = re.sub(r"\s*``\s*", " ", text)
    text = re.sub(r"\s*''\s*", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    # Remove 3+ consecutive blank lines down to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class LLMClient:
    """Unified LLM client interface with automatic multi-model failover."""

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

        # Active Groq models in fallback priority order
        self.groq_models = [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ]

        self._groq_client = None
        self._gemini_client = None

        # Initialize Groq client
        if self.groq_api_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=self.groq_api_key, max_retries=0)
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")

        # Initialize Gemini client
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
        temperature: float = 0.15,
        max_tokens: int = settings.MAX_OUTPUT_TOKENS,
    ) -> str:
        """Generate response with automatic retry and model fallback."""
        # temperature=0.15 gives natural language variation while staying factually grounded
        # 1. Attempt generation using Groq models
        if self._groq_client:
            for model_name in self.groq_models:
                for attempt in range(2):
                    try:
                        res = self._generate_groq(prompt, system_instruction, temperature, max_tokens, model=model_name)
                        return clean_output_formatting(res)
                    except Exception as e:
                        err_msg = str(e).lower()
                        if "429" in err_msg or "rate limit" in err_msg:
                            logger.info(f"Groq {model_name} rate limit on attempt {attempt+1}, waiting 3.5s...")
                            time.sleep(3.5)
                        else:
                            break

        # 2. Fallback to Google Gemini
        if self._gemini_client:
            try:
                res = self._generate_gemini(prompt, system_instruction, temperature, max_tokens)
                return clean_output_formatting(res)
            except Exception as e:
                logger.warning(f"Gemini generation failed: {e}.")

        # 3. Final retry on primary Groq model after brief cooldown
        if self._groq_client:
            time.sleep(4.0)
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
        """Execute chat completion request via Groq API."""
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
        """Execute content generation request via Google GenAI API."""
        full_content = f"{system_instruction}\n\nUser Question:\n{prompt}"
        response = self._gemini_client.models.generate_content(
            model=self.gemini_model,
            contents=full_content,
        )
        return response.text.strip()


def _detect_question_type(query: str) -> str:
    """Classify query intent to guide answer structure."""
    q = query.lower().strip()

    # Procedural / flow patterns
    if any(kw in q for kw in ["procedure", "flow", "how does", "how do", "explain the", "establishment", "registration", "handover", "sequence", "steps", "process"]):
        return "procedural"

    # Comparison / multi-part patterns
    if any(kw in q for kw in ["difference", "compare", "vs", "types of", "list", "what are the", "core functions", "main functions", "key features", "roles"]):
        return "comparison"

    # Broad / overview patterns
    if any(kw in q for kw in ["tell me about", "overview", "explain 5g", "how 5g", "what is 5g", "network working", "architecture"]):
        return "overview"

    # Default: conceptual
    return "conceptual"


def format_grounded_prompt(
    query: str,
    retrieved_chunks: List[RetrievalResult],
) -> str:
    """Format user prompt with context excerpts and adaptive structure guidance."""
    context_blocks = ["=== 3GPP SPECIFICATION CONTEXT EXCERPTS ==="]
    for idx, chunk in enumerate(retrieved_chunks, 1):
        clean_chunk_text = chunk.text.strip()
        # Keep chunk text compact to stay within token budgets
        if len(clean_chunk_text) > 900:
            clean_chunk_text = clean_chunk_text[:900] + "..."

        context_blocks.append(
            f"--- Context Source [{idx}] ---\n"
            f"Specification: {chunk.document_code} (Rel-{chunk.release})\n"
            f"Clause: {chunk.section_number} - {chunk.section_title}\n"
            f"Page Number: {chunk.page_number}\n"
            f"Content:\n{clean_chunk_text}\n"
        )

    context_str = "\n".join(context_blocks)

    # Choose structure guidance based on detected question type
    q_type = _detect_question_type(query)

    if q_type == "procedural":
        style_hint = (
            "QUESTION TYPE: PROCEDURAL / FLOW.\n"
            "Write exactly: (1) One intro sentence saying what this procedure does and what triggers it, with one inline citation. "
            "(2) 4-6 numbered steps describing the actual message exchange from the excerpts. "
            "CRITICAL: Only include steps that are explicitly described in the context above. "
            "Do NOT fill gaps by guessing or adding steps from general 3GPP knowledge. "
            "If only part of the procedure is covered, describe only that part and note it is a subset. "
            "Keep each step to one plain sentence. Total answer: 200-280 words."
        )
    elif q_type == "comparison":
        style_hint = (
            "QUESTION TYPE: COMPONENT / FUNCTION LIST.\n"
            "Write a clear 3-4 sentence intro paragraph explaining what this component is and its role in the 5G Core architecture. "
            "Then use 2-3 H4 headings (####) grouping related capabilities. "
            "Each heading ends with its citation e.g. #### NAS Termination & Registration [TS 23.501 Clause 6.2.1, Page 423]. "
            "Under each heading: 2-3 bullets that explain the capabilities in plain language — do not copy verbatim spec text. "
            "Total answer: 220-290 words."
        )
    elif q_type == "overview":
        style_hint = (
            "QUESTION TYPE: BROAD OVERVIEW.\n"
            "Write 2 focused prose paragraphs. First paragraph: describe the overall system components and how they connect. "
            "Second paragraph: describe 2-3 key capabilities or design principles that make 5G distinct. "
            "Be selective — use only the most important points from the excerpts. Cite inline once per paragraph. "
            "Do NOT try to mention every detail. Total answer: 160-240 words."
        )
    else:  # conceptual
        style_hint = (
            "QUESTION TYPE: CONCEPTUAL.\n"
            "Write 2-3 prose paragraphs explaining the concept, its purpose in the 5G architecture, and how it works in practice. "
            "Keep a natural conversational tone — explain as if to a colleague who knows 5G basics but not this specific feature. "
            "Cite inline once per paragraph. Do not paste verbatim spec sentences. "
            "Total answer: 180-260 words."
        )

    user_prompt = (
        f"{context_str}\n\n"
        f"=== USER QUESTION ===\n{query}\n\n"
        f"{style_hint}"
    )
    return user_prompt
