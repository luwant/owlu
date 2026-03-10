"""LLM-based phrase extraction using DeepSeek-compatible API."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Iterable

from ..common.types import CandidatePhrase, OWLUConfig


class LLMOutputError(ValueError):
    """Raised when LLM response cannot be parsed into expected JSON schema."""


def get_api_key(env_var: str = "DEEPSEEK_API_KEY") -> str:
    api_key = os.getenv(env_var, "").strip()
    if not api_key:
        raise EnvironmentError(f"Missing required environment variable: {env_var}")
    return api_key


class LLMPhraseGenerator:
    """Generate phrase candidates from documents using LLM."""

    def __init__(self, config: OWLUConfig, client: Any | None = None):
        self.config = config
        self.client = client or self._build_default_client()

    def _build_default_client(self) -> Any:
        from openai import OpenAI  # type: ignore

        api_key = self.config.llm_api_key or get_api_key()
        return OpenAI(
            api_key=api_key,
            base_url=self.config.llm_base_url,
            timeout=self.config.llm_timeout_seconds,
        )

    def _system_prompt(self) -> str:
        return (
            "You extract high-value label phrases from text. "
            "Output JSON with keys: summary, phrases, evidence."
        )

    def _user_prompt(self, text: str) -> str:
        max_phrases = self.config.llm_max_phrases
        return (
            "Extract concise noun phrases that can be labels.\n"
            f"Return 1-{max_phrases} phrases only.\n"
            "Keep phrases short (2-6 words), avoid generic terms.\n"
            "Return strict json with keys: summary, phrases, evidence.\n\n"
            f"Text:\n{text}"
        )

    def _extract_json_payload(self, content: str) -> dict[str, Any]:
        text = (content or "").strip()
        if not text:
            raise LLMOutputError("Empty LLM response content")

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise LLMOutputError("No JSON object found in LLM response")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMOutputError(f"Invalid JSON in LLM response: {exc}") from exc

    def _request_once(self, text: str) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": self._user_prompt(text)},
            ],
        )
        content = response.choices[0].message.content
        if content is None:
            raise LLMOutputError("LLM response content is null")
        return self._extract_json_payload(content)

    def _build_candidates(
        self,
        payload: dict[str, Any],
        doc_id: str,
        pass_id: int,
        agreement_map: dict[str, float] | None = None,
    ) -> list[CandidatePhrase]:
        summary = payload.get("summary")
        phrases = payload.get("phrases")
        evidence = payload.get("evidence")

        if isinstance(phrases, str):
            phrases = [p.strip() for p in phrases.split(",") if p.strip()]
        if not isinstance(phrases, list):
            raise LLMOutputError(
                "JSON field 'phrases' must be a list (or comma-separated string)"
            )
        if evidence is None:
            evidence = []
        elif isinstance(evidence, str):
            evidence = [evidence.strip()] if evidence.strip() else []
        elif not isinstance(evidence, list):
            evidence = [str(evidence)]

        candidates: list[CandidatePhrase] = []
        seen: set[str] = set()
        for raw in phrases:
            if not isinstance(raw, str):
                continue
            phrase = " ".join(raw.strip().split())
            if not phrase:
                continue
            key = phrase.lower()
            if key in seen:
                continue
            seen.add(key)
            if len(candidates) >= self.config.llm_max_phrases:
                break
            agreement = (
                1.0 if agreement_map is None else float(agreement_map.get(key, 0.0))
            )
            candidates.append(
                CandidatePhrase(
                    text=phrase,
                    raw_text=raw,
                    source_doc_id=doc_id,
                    timestamp=datetime.now(timezone.utc),
                    summary=summary if isinstance(summary, str) else None,
                    evidence=[e for e in evidence if isinstance(e, str)],
                    agreement=agreement,
                    pass_id=pass_id,
                    source_count=1,
                )
            )
        return candidates

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, text: str, doc_id: str) -> list[CandidatePhrase]:
        """Single-pass phrase extraction."""
        payload = self._request_once(text)
        return self._build_candidates(payload=payload, doc_id=doc_id, pass_id=1)

    def should_trigger_uncertain(
        self, top1_score: float, top2_score: float
    ) -> bool:
        """Decide whether uncertainty-based multi-sampling should kick in."""
        if top1_score < self.config.uncertain_top1_threshold:
            return True
        margin = top1_score - top2_score
        return margin < self.config.uncertain_margin_threshold

    def generate_uncertain_batch(
        self,
        texts: list[str],
        doc_ids: list[str],
        scores: Iterable[tuple[float, float]],
    ) -> list[CandidatePhrase]:
        results: list[CandidatePhrase] = []
        for text, doc_id, (top1, top2) in zip(texts, doc_ids, scores):
            if self.should_trigger_uncertain(top1, top2):
                results.extend(
                    self.multi_sample_aggregate(text=text, doc_id=doc_id)
                )
        return results

    def multi_sample_aggregate(
        self, text: str, doc_id: str, k: int | None = None
    ) -> list[CandidatePhrase]:
        """k-shot sampling with consistency voting."""
        samples = int(k or self.config.multi_sample_k)
        if samples <= 0:
            raise ValueError("k must be > 0")

        payloads = [self._request_once(text) for _ in range(samples)]
        phrase_votes: dict[str, int] = {}
        phrase_raw: dict[str, str] = {}
        summary = None
        evidence: list[str] = []

        for payload in payloads:
            if summary is None and isinstance(payload.get("summary"), str):
                summary = payload["summary"]
            if not evidence and isinstance(payload.get("evidence"), list):
                evidence = [
                    e for e in payload["evidence"] if isinstance(e, str)
                ]
            phrases = payload.get("phrases", [])
            if not isinstance(phrases, list):
                continue
            for raw in phrases:
                if not isinstance(raw, str):
                    continue
                phrase = " ".join(raw.strip().split())
                if not phrase:
                    continue
                key = phrase.lower()
                phrase_votes[key] = phrase_votes.get(key, 0) + 1
                phrase_raw.setdefault(key, phrase)

        if not phrase_votes:
            raise LLMOutputError(
                "No valid phrases found across multi-sample aggregation"
            )

        sorted_keys = sorted(phrase_votes, key=lambda x: (-phrase_votes[x], x))
        top_keys = sorted_keys[: self.config.llm_max_phrases]
        agreement_map = {
            key: phrase_votes[key] / float(samples) for key in top_keys
        }

        payload = {
            "summary": summary,
            "phrases": [phrase_raw[key] for key in top_keys],
            "evidence": evidence,
        }
        return self._build_candidates(
            payload=payload, doc_id=doc_id, pass_id=2, agreement_map=agreement_map
        )
