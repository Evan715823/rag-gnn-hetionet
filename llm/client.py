from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import anthropic

from llm.prompts import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)

DEFAULT_MODEL = "claude-sonnet-4-6"


@dataclass
class LLMResponse:
    prediction: str
    confidence: float
    rationale: str
    raw: str


def _parse_json(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    return json.loads(m.group(0))


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Put it in environment or pass api_key=."
            )
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model

    def predict(self, compound_name: str, compound_id: str,
                disease_name: str, disease_id: str,
                paths_block: str, max_tokens: int = 600) -> LLMResponse:
        user_text = USER_PROMPT_TEMPLATE.format(
            compound_name=compound_name,
            compound_id=compound_id,
            disease_name=disease_name,
            disease_id=disease_id,
            paths_block=paths_block,
        )
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_text}],
        )
        raw = resp.content[0].text
        try:
            parsed = _parse_json(raw)
        except Exception:
            return LLMResponse(prediction="no", confidence=0.0, rationale="(parse error)", raw=raw)
        return LLMResponse(
            prediction=str(parsed.get("prediction", "no")).lower(),
            confidence=float(parsed.get("confidence", 0.0)),
            rationale=str(parsed.get("rationale", "")),
            raw=raw,
        )

    def judge_faithfulness(self, paths_block: str, rationale: str, max_tokens: int = 400) -> dict:
        user_text = JUDGE_USER_PROMPT_TEMPLATE.format(
            paths_block=paths_block,
            rationale=rationale,
        )
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=[{
                "type": "text",
                "text": JUDGE_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_text}],
        )
        raw = resp.content[0].text
        try:
            return _parse_json(raw)
        except Exception:
            return {"faithful": False, "invented_entities": [], "explanation": "parse error", "raw": raw}
