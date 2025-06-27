# paperqa_text_processor.py
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Optional, Tuple, Dict

import paperqa
from paperqa import Settings
from paperqa.settings import AgentSettings

from mdsgene.cache_utils import save_formatted_result
from mdsgene.internal.defines import NO_INFORMATION_LIST

# --- Configuration ---
# It's best practice to load API keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Default model - this can be overridden in constructor
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


class PaperQATextProcessor:
    """Handles general text interactions with the paper-qa library (e.g., formatting)."""

    def __init__(
        self,
        model_name: str = DEFAULT_GEMINI_MODEL,
        *,
        pmid: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initializes the PaperQA client for text operations.

        Args:
            model_name: The specific model to use with paper-qa.
            pmid: PMID for organizing formatted answer cache.
            api_key: Google AI API key. If None, uses GEMINI_API_KEY env var.
        """
        resolved_api_key = api_key or GEMINI_API_KEY
        if not resolved_api_key:
            raise ValueError("Gemini API key not provided and GEMINI_API_KEY environment variable not set.")

        try:
            # Configure PaperQA with Gemini models using Settings. Docs objects
            # are immutable, so we pass the models and embedding here rather
            # than mutating attributes after creation.
            self.settings = Settings(
                llm=f"gemini/{model_name}",
                summary_llm=f"gemini/{model_name}",
                agent=AgentSettings(agent_llm=f"gemini/{model_name}"),
                embedding="gemini/text-embedding-004"
            )

            # Initialize paper-qa using our settings (no PDF directory needed)
            self.docs = paperqa.Docs(settings=self.settings)
            self.model_name = model_name
            print(f"PaperQATextProcessor initialized with Gemini model '{model_name}'.")

            self.pmid = pmid
            cache_dir = Path("cache") / pmid if pmid else Path("cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._format_cache_path = cache_dir / "formatted_answer_cache.json"
            self._format_cache: Dict[str, Dict[str, str]] = {}

            # Load existing cache if available
            if self._format_cache_path.exists():
                try:
                    with open(self._format_cache_path, "r", encoding="utf-8") as f:
                        self._format_cache = json.load(f)
                    print(f"Loaded {len(self._format_cache)} cached formatted answers.")
                except Exception as e:
                    print(f"WARNING: Failed to load format cache: {e}")
                    self._format_cache = {}

        except Exception as e:
            print(f"Error initializing PaperQATextProcessor: {e}", file=sys.stderr)
            traceback.print_exc()
            raise

    def load_from_cache(self, cache_id: str) -> Optional[Dict[str, str]]:
        """Load a cached formatted answer by id."""
        return self._format_cache.get(cache_id)

    def save_to_cache(self, cache_id: str, formatted_answer: str, raw_answer: str, strategy: str) -> None:
        """Save a formatted answer to cache."""
        self._format_cache[cache_id] = {
            "formatted_answer": formatted_answer,
            "raw_answer": raw_answer,
            "strategy": strategy
        }
        try:
            with open(self._format_cache_path, "w", encoding="utf-8") as f:
                json.dump(self._format_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"ERROR saving format cache: {e}")

    def format_answer(self, raw_answer: Optional[str], strategy: str) -> Tuple[str, Optional[str]]:
        """
        Uses PaperQA to format a raw answer according to a specific strategy.
        Uses caching to avoid repeated formatting for the same raw_answer and strategy.

        Args:
            raw_answer: The raw answer text to format.
            strategy: The formatting strategy to apply.

        Returns:
            A tuple of (formatted_answer, context).
        """
        print("  Formatting answer using PaperQATextProcessor...")
        if raw_answer.lower() in NO_INFORMATION_LIST:
            print(f"  Raw answer indicates missing info ('{raw_answer}'). Returning -99.")
            return "-99", None
        if "error" in raw_answer.lower() or "failed" in raw_answer.lower():
            print(f"  Raw answer indicates error ('{raw_answer[:50]}...'). Returning -99.")
            return "-99", None

        # Create a cache key based on raw_answer and strategy
        cache_key = f"{hash(raw_answer)}-{hash(strategy)}"

        # Check if we have a cached result
        if cache_key in self._format_cache:
            cached_data = self._format_cache[cache_key]
            print("  Using cached formatted answer for this raw_answer/strategy combination.")
            return cached_data.get("formatted_answer", "-99"), raw_answer

        formatting_prompt = f"""
Given the following raw text extracted from a document and a desired formatting strategy, reformat the text.

Raw Text:
\"\"\"
{raw_answer}
\"\"\"

Formatting Strategy / Expected Output:
\"\"\"
{strategy}
\"\"\"

Formatting Rules:
1.  Follow the strategy precisely.
2.  Numeric answers (age, counts): Extract ONLY the number. If unknown/not given/not found in raw text, output -99. If explicitly zero/none, output 0.
3.  Yes/no questions: Output ONLY lowercase "yes" or "no". If unknown/not mentioned/unclear in raw text, output -99.
4.  Sex: Output ONLY uppercase "M" or "F". If unknown/not reported, output -99.
5.  Zygosity: Output ONLY "heterozygous", "homozygous", or "hemizygous". If unknown/not applicable, output -99.
6.  Inclusion decisions (IN/EX): Output ONLY uppercase "IN" or "EX". If unclear, output -99.
7.  HGNC Gene Symbols: Output the official symbol if found (e.g., "PARK2"). If multiple possibilities, list them comma-separated unless strategy specifies otherwise. If none found or not applicable, output -99.
8.  Mutation Notation (cDNA/protein): Output the notation as described in the strategy (e.g., c.511C>T, p.Gln171*). Standardize if possible, but preserve original if instructed. If none found/not applicable, output -99.
9.  General Text/Comments: Extract the relevant information as described. If none found, output -99.
10. IMPORTANT: Output ONLY the final formatted value. Do NOT include explanations, apologies, or any text like "Formatted answer:", "Based on the text:", "The value is:", etc. JUST the value or -99.
11. If the Raw Text explicitly states the information is missing (e.g., "not reported", "unknown", "N/A"), output -99.

Formatted Value:"""

        try:
            # Query paper-qa for formatting
            answer = self.docs.query(formatting_prompt)
            formatted_answer = str(answer)

            if not formatted_answer:
                print(
                    f"  ERROR: PaperQA formatting failed for raw answer: '{raw_answer[:50]}...'. Returning -99.",
                    file=sys.stderr,
                )
                # Return the original raw_answer as context even on failure
                return "-99", raw_answer

            if not formatted_answer or not formatted_answer.strip():
                print("ERROR: Empty string returned from PaperQA.")
                return "-99", raw_answer

            # Post-processing cleanup
            formatted_answer = formatted_answer.strip()
            if formatted_answer.startswith('"') and formatted_answer.endswith('"') and len(formatted_answer) > 1:
                formatted_answer = formatted_answer[1:-1].strip()
            elif formatted_answer.startswith("'") and formatted_answer.endswith("'") and len(formatted_answer) > 1:
                formatted_answer = formatted_answer[1:-1].strip()

            preambles = ["formatted value:", "formatted answer:", "the formatted value is:", "output:"]
            for preamble in preambles:
                if formatted_answer.lower().startswith(preamble):
                    formatted_answer = formatted_answer[len(preamble):].strip()
                    break

            if not formatted_answer or formatted_answer.lower() in NO_INFORMATION_LIST:
                print(f"  Formatted answer resulted in an 'unknown' value ('{formatted_answer}'). Returning -99.")
                # Return the original raw_answer as context
                return "-99", raw_answer

            if self.pmid:
                text_for_parse = formatted_answer.strip()
                if text_for_parse.startswith("```"):
                    text_for_parse = re.sub(r"^```(?:json)?\s*", "", text_for_parse)
                    text_for_parse = re.sub(r"\s*```$", "", text_for_parse).strip()
                try:
                    parsed_json = json.loads(text_for_parse)
                except Exception:
                    parsed_json = None

                if isinstance(parsed_json, dict):
                    save_formatted_result(
                        self.pmid,
                        formatting_prompt,
                        raw_answer,
                        strategy,
                        parsed_json,
                        "paperqa",
                    )
                else:
                    print("[Cache] Formatted result is not JSON. Skipping cache save.")

            print(f"  Formatted result: '{formatted_answer}'")

            # Save to cache for future use
            cache_key = f"{hash(raw_answer)}-{hash(strategy)}"
            self.save_to_cache(cache_key, formatted_answer, raw_answer, strategy)
            print("  Cached formatted answer for future use.")

            # Return the formatted answer and the original raw_answer as context
            return formatted_answer, raw_answer

        except Exception as e:
            print(f"ERROR: Failed to format answer: {e}", file=sys.stderr)
            traceback.print_exc()
            return "-99", raw_answer
