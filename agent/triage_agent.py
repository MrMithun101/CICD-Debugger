import json
import logging
import os
from dataclasses import dataclass, asdict

from cerebras.cloud.sdk import Cerebras

from ingestion.cleaner import clean_log
from ingestion.classifier import classify_failure
from ingestion.chunker import chunk_log
from vector_store.chroma_store import ChromaStore
from agent.prompts import build_system_prompt, build_user_prompt

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama3.1-8b"


@dataclass
class TriageReport:
    failure_type: str
    root_cause: str
    confidence: float
    suggested_fix: str
    explanation: str
    similar_cases: list[dict]
    log_preview: str    # first ~500 chars of the cleaned log for display

    def to_dict(self) -> dict:
        return asdict(self)


class TriageAgent:
    def __init__(
        self,
        store: ChromaStore,
        cerebras_api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ):
        self.store = store
        self.model = model
        self.client = Cerebras(api_key=cerebras_api_key or os.environ["CEREBRAS_API_KEY"])

    def triage(self, log_text: str, n_similar: int = 3) -> TriageReport:
        """Analyze a failing CI/CD log and return a structured triage report.

        Steps:
          1. Clean and classify the log
          2. Extract the most failure-dense snippet for the prompt
          3. Retrieve similar historical failures from ChromaDB
          4. Call Cerebras to generate root cause, fix, and explanation
        """
        cleaned = clean_log(log_text)
        failure_type = classify_failure(cleaned)

        chunks = chunk_log(cleaned)
        snippet = self._extract_relevant_snippet(chunks)

        # Try type-filtered retrieval first; fall back to unfiltered if empty.
        similar = self.store.query(snippet, n_results=n_similar, failure_type=failure_type)
        if not similar:
            similar = self.store.query(snippet, n_results=n_similar)

        llm_result = self._call_cerebras(snippet, similar, failure_type)

        return TriageReport(
            failure_type=failure_type,
            root_cause=llm_result.get("root_cause", "Unable to determine root cause"),
            confidence=float(llm_result.get("confidence", 0.0)),
            suggested_fix=llm_result.get("suggested_fix", "No fix suggested"),
            explanation=llm_result.get("explanation", ""),
            similar_cases=similar,
            log_preview=cleaned[:500],
        )

    def _extract_relevant_snippet(self, chunks: list[dict]) -> str:
        """Return the most failure-relevant chunk text.

        Failures appear near the end of logs (after setup completes), so we
        scan from the last chunk backwards and take the first non-trivial one.
        """
        if not chunks:
            return ""
        for chunk in reversed(chunks):
            if len(chunk["text"].strip()) > 50:
                return chunk["text"][:3000]
        return chunks[-1]["text"][:3000]

    def _call_cerebras(
        self,
        log_snippet: str,
        similar_cases: list[dict],
        failure_type: str,
    ) -> dict:
        """Call Cerebras with JSON mode and return the parsed response dict."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": build_system_prompt()},
                    {"role": "user", "content": build_user_prompt(log_snippet, similar_cases, failure_type)},
                ],
                response_format={"type": "json_object"},
                max_tokens=1024,
                temperature=0.1,
            )
            content = response.choices[0].message.content
            return json.loads(content)

        except json.JSONDecodeError as e:
            logger.warning("Failed to parse Cerebras JSON response: %s", e)
            return {
                "root_cause": "Could not parse model response",
                "confidence": 0.0,
                "suggested_fix": "",
                "explanation": "",
            }
        except Exception as e:
            logger.error("Cerebras API call failed: %s", e)
            return {
                "root_cause": f"API error: {e}",
                "confidence": 0.0,
                "suggested_fix": "",
                "explanation": "",
            }
