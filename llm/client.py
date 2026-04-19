from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import requests

from llm.prompts import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)

DEFAULT_MODEL = "grok-4-fast-reasoning"
API_URL = "https://api.x.ai/v1/chat/completions"


@dataclass
class LLMResponse:
    prediction: str
    confidence: float
    rationale: str
    raw: str


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    return json.loads(m.group(0))


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None, timeout: int = 180):
        key = api_key or os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
        if not key:
            raise RuntimeError(
                "XAI_API_KEY not set. Put it in environment (XAI_API_KEY=xai-...) or pass api_key=."
            )
        self.api_key = key
        self.model = model
        self.timeout = timeout

    def _call(self, system: str, user: str, max_tokens: int = 600) -> str:
        resp = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.0,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

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
        raw = self._call(SYSTEM_PROMPT, user_text, max_tokens)
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
        raw = self._call(JUDGE_SYSTEM_PROMPT, user_text, max_tokens)
        try:
            return _parse_json(raw)
        except Exception:
            return {"faithful": False, "invented_entities": [], "explanation": "parse error", "raw": raw}
