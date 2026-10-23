from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

DEFAULT_LAYA_MODEL = "convaiinnovations/laya-typed-decisions"


@dataclass
class LayaConfig:
    enabled: bool = False
    model: str = DEFAULT_LAYA_MODEL

    @classmethod
    def from_env_and_args(
        cls,
        cli_laya: Optional[str] = None,
        cli_model: Optional[str] = None,
    ) -> "LayaConfig":
        """Resolves configuration with precedence: CLI flag > Environment variable > Default (off)."""
        # Precedence: CLI flag wins over JEV_LAYA env var
        if cli_laya is not None:
            enabled = cli_laya.strip().lower() == "on"
        elif "JEV_LAYA" in os.environ:
            val = os.environ["JEV_LAYA"].strip().lower()
            enabled = val in ("1", "true", "on", "yes")
        else:
            enabled = False

        model = (cli_model or os.environ.get("JEV_LAYA_MODEL") or DEFAULT_LAYA_MODEL).strip()
        return cls(enabled=enabled, model=model)


@dataclass
class LayaResult:
    verdict: Optional[bool] = None
    probability: Optional[float] = None
    reason: Optional[str] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "probability": self.probability,
            "reason": self.reason,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


class LayaShadowJudge:
    """Secondary (shadow) judge evaluating attempts independently using Laya.

    Advisory only:
    - Never blocks, retries, or alters gate outcomes.
    - Receives same inputs as Jev without seeing Jev's verdict (no anchoring).
    - Fails open on timeout, network error, or missing dependency.
    - API credentials read strictly from environment variables.
    """

    def __init__(self, config: Optional[LayaConfig] = None):
        self.config = config or LayaConfig()
        self._agent: Optional[Any] = None
        self._init_failed: bool = False
        self._warned_error: bool = False

    @property
    def enabled(self) -> bool:
        return self.config.enabled and not self._init_failed

    def _get_agent(self) -> Optional[Any]:
        if not self.config.enabled or self._init_failed:
            return None
        if self._agent is None:
            try:
                import laya

                self._agent = laya.load(self.config.model)
            except Exception as exc:
                self._init_failed = True
                if not self._warned_error:
                    logger.warning(
                        f"[LAYA] Failed to initialize Laya model '{self.config.model}': {exc}. "
                        "Continuing with Laya shadow evaluation disabled (fail-open)."
                    )
                    self._warned_error = True
                return None
        return self._agent

    def evaluate(
        self,
        subgoal_description: str,
        subgoal_scope: Any,
        diff: str,
        investigation_notes: Optional[str] = None,
        mechanical_detail: str = "",
        dependency_delta: Optional[Dict[str, Any]] = None,
    ) -> LayaResult:
        """Evaluates diff independently without seeing Jev's verdict (unanchored)."""
        if not self.config.enabled:
            return LayaResult()

        agent = self._get_agent()
        if agent is None:
            return LayaResult(error="Laya agent unavailable")

        t0 = time.perf_counter()
        try:
            state: Dict[str, Any] = {
                "subgoal": subgoal_description,
                "description": subgoal_description,
                "scope": subgoal_scope,
                "diff": diff,
            }
            if investigation_notes:
                state["investigation_notes"] = investigation_notes
            if mechanical_detail:
                state["mechanical_detail"] = mechanical_detail
            if dependency_delta:
                state["dependency_delta"] = dependency_delta

            questions = {
                "valid": {
                    "type": "noul",
                    "instructions": (
                        "Does the diff fully and correctly implement the subgoal described in state, "
                        "without exceeding its declared scope?"
                    ),
                }
            }

            res = agent.predict(state=state, questions=questions)
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)

            answers = res.get("answers", {}) if isinstance(res, dict) else {}
            valid_ans = answers.get("valid", {}) if isinstance(answers, dict) else {}
            noul_val = valid_ans.get("noul")
            reason = valid_ans.get("explanation") or valid_ans.get("reason")

            if noul_val is not None:
                prob = float(noul_val)
                return LayaResult(
                    verdict=(prob >= 0.5),
                    probability=prob,
                    reason=str(reason) if reason else None,
                    latency_ms=latency_ms,
                )
            return LayaResult(
                error="Malformed response: missing noul probability",
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            if not self._warned_error:
                logger.warning(f"[LAYA] Shadow evaluation error: {exc}. Continuing fail-open.")
                self._warned_error = True
            return LayaResult(error=str(exc), latency_ms=latency_ms)
