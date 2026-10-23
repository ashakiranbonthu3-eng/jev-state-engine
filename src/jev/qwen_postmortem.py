import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

from jev.ledger import get_jev_home

logger = logging.getLogger(__name__)

DEFAULT_QWEN_URL = "http://127.0.0.1:8080"
DEFAULT_QWEN_TIMEOUT = 120.0
DEFAULT_QWEN_MAX_DIFF_CHARS = 24000
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.2


@dataclass
class QwenConfig:
    enabled: bool = False
    url: str = DEFAULT_QWEN_URL
    model: Optional[str] = None
    timeout: float = DEFAULT_QWEN_TIMEOUT
    max_diff_chars: int = DEFAULT_QWEN_MAX_DIFF_CHARS
    feedback: bool = False

    @classmethod
    def from_env_and_args(
        cls,
        cli_postmortem: Optional[str] = None,
        cli_url: Optional[str] = None,
        cli_model: Optional[str] = None,
        cli_timeout: Optional[Union[int, float]] = None,
        cli_max_diff_chars: Optional[int] = None,
        cli_feedback: Optional[str] = None,
    ) -> "QwenConfig":
        """Resolves configuration with precedence: CLI flag > env var > default."""

        # 1. Enabled: --qwen-postmortem on|off, JEV_QWEN_POSTMORTEM, default off
        if cli_postmortem is not None:
            enabled = cli_postmortem.strip().lower() == "on"
        elif "JEV_QWEN_POSTMORTEM" in os.environ:
            val = os.environ["JEV_QWEN_POSTMORTEM"].strip().lower()
            enabled = val in ("1", "true", "on", "yes")
        else:
            enabled = False

        # 2. Server URL: --qwen-url, JEV_QWEN_URL, default http://127.0.0.1:8080
        if cli_url is not None and cli_url.strip():
            url = cli_url.strip()
        elif "JEV_QWEN_URL" in os.environ and os.environ["JEV_QWEN_URL"].strip():
            url = os.environ["JEV_QWEN_URL"].strip()
        else:
            url = DEFAULT_QWEN_URL

        # 3. Model: --qwen-model, JEV_QWEN_MODEL, default None (auto-detect)
        if cli_model is not None and cli_model.strip():
            model = cli_model.strip()
        elif "JEV_QWEN_MODEL" in os.environ and os.environ["JEV_QWEN_MODEL"].strip():
            model = os.environ["JEV_QWEN_MODEL"].strip()
        else:
            model = None

        # 4. Timeout: --qwen-timeout, JEV_QWEN_TIMEOUT, default 120
        if cli_timeout is not None:
            timeout = float(cli_timeout)
        elif "JEV_QWEN_TIMEOUT" in os.environ and os.environ["JEV_QWEN_TIMEOUT"].strip():
            try:
                timeout = float(os.environ["JEV_QWEN_TIMEOUT"].strip())
            except ValueError:
                timeout = DEFAULT_QWEN_TIMEOUT
        else:
            timeout = DEFAULT_QWEN_TIMEOUT

        # 5. Max diff chars: --qwen-max-diff-chars, JEV_QWEN_MAX_DIFF_CHARS, default 24000
        if cli_max_diff_chars is not None:
            max_diff_chars = int(cli_max_diff_chars)
        elif "JEV_QWEN_MAX_DIFF_CHARS" in os.environ and os.environ["JEV_QWEN_MAX_DIFF_CHARS"].strip():
            try:
                max_diff_chars = int(os.environ["JEV_QWEN_MAX_DIFF_CHARS"].strip())
            except ValueError:
                max_diff_chars = DEFAULT_QWEN_MAX_DIFF_CHARS
        else:
            max_diff_chars = DEFAULT_QWEN_MAX_DIFF_CHARS

        # 6. Feedback: --qwen-feedback on|off, JEV_QWEN_FEEDBACK, default off
        if cli_feedback is not None:
            feedback = cli_feedback.strip().lower() == "on"
        elif "JEV_QWEN_FEEDBACK" in os.environ:
            val = os.environ["JEV_QWEN_FEEDBACK"].strip().lower()
            feedback = val in ("1", "true", "on", "yes")
        else:
            feedback = False

        return cls(
            enabled=enabled,
            url=url,
            model=model,
            timeout=timeout,
            max_diff_chars=max_diff_chars,
            feedback=feedback,
        )


@dataclass
class QwenPostmortemResult:
    chain_of_thought: str
    where_it_went_wrong: str
    root_cause: str
    suggested_fix: str
    confidence: str  # low | medium | high
    raw_text: Optional[str] = None
    latency_ms: float = 0.0
    error: Optional[str] = None
    parse_fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_of_thought": self.chain_of_thought,
            "where_it_went_wrong": self.where_it_went_wrong,
            "root_cause": self.root_cause,
            "suggested_fix": self.suggested_fix,
            "confidence": self.confidence,
            "raw_text": self.raw_text,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "parse_fallback": self.parse_fallback,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


def format_postmortem_prompt(
    ticket: str,
    rejecting_source: str,
    rejection_reason: str,
    jev_verdict: Optional[Any],
    diff: str,
    strike_number: str,
    previous_strikes: Optional[List[Dict[str, Any]]] = None,
    max_diff_chars: int = DEFAULT_QWEN_MAX_DIFF_CHARS,
) -> str:
    """Constructs a deterministic prompt for Qwen post-mortem analysis."""
    # 1. Truncate diff with explicit marker if needed
    if len(diff) > max_diff_chars:
        truncated_diff = diff[:max_diff_chars] + "\n... [TRUNCATED] ...\n"
    else:
        truncated_diff = diff

    # 2. Truncate error/rejection output if extremely large
    max_err_chars = 16000
    if len(rejection_reason) > max_err_chars:
        cleaned_reason = rejection_reason[:max_err_chars] + "\n... [TRUNCATED] ...\n"
    else:
        cleaned_reason = rejection_reason

    # 3. Format Jev output (say "no reason provided" if null)
    if jev_verdict is not None:
        verdict_bool = getattr(jev_verdict, "valid", None)
        verdict_str = "Valid" if verdict_bool is True else "Invalid"
        reason_val = getattr(jev_verdict, "reason", None)
        if reason_val is not None and str(reason_val).strip():
            reason_str = str(reason_val).strip()
        else:
            reason_str = "no reason provided"
        jev_output_text = f"Verdict: {verdict_str}\nReason: {reason_str}"
    else:
        jev_output_text = "Verdict: Not evaluated (failed prior check)\nReason: no reason provided"

    # 4. Format previous strikes
    if not previous_strikes:
        prev_strikes_text = "None (first strike for this ticket)"
    else:
        items = []
        for s in previous_strikes:
            s_no = s.get("strike_no", "?")
            s_src = s.get("source", "Unknown")
            s_reason = s.get("reason", "")
            if len(s_reason) > 200:
                s_reason = s_reason[:200] + "..."
            s_rc = s.get("root_cause", "")
            items.append(f"- Strike {s_no}: Source: {s_src} | Reason: {s_reason} | Root Cause: {s_rc}")
        prev_strikes_text = "\n".join(items)

    prompt = (
        "You are an advisory post-mortem analyzer for an autonomous coding agent.\n"
        "Analyze why the following implementation attempt failed its verification gate.\n\n"
        f"TICKET:\n{ticket}\n\n"
        f"STRIKE:\n{strike_number}\n\n"
        f"PREVIOUS STRIKES:\n{prev_strikes_text}\n\n"
        f"REJECTING VERDICT:\nSource: {rejecting_source}\nReason / Error Output:\n{cleaned_reason}\n\n"
        f"JEV OUTPUT:\n{jev_output_text}\n\n"
        f"GIT DIFF:\n{truncated_diff}\n\n"
        "Respond with a valid JSON object strictly matching this schema:\n"
        "{\n"
        '  "chain_of_thought": "step-by-step reasoning about what the worker was trying to do and where the attempt diverged from the ticket or broke a check",\n'
        '  "where_it_went_wrong": "specific files/lines/hunks from the diff, tied to the rejection reason",\n'
        '  "root_cause": "one short statement",\n'
        '  "suggested_fix": "concrete direction for the next attempt, no full code",\n'
        '  "confidence": "low" | "medium" | "high"\n'
        "}\n"
        "Do not include any markdown fences or commentary outside the JSON.\n"
    )
    return prompt


def parse_qwen_response(content: str) -> Tuple[Dict[str, Any], bool]:
    """Parses Qwen JSON response with fallback to raw text if parsing fails."""
    cleaned = content.strip()
    # Strip markdown fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            # Validate and normalize expected fields
            confidence = str(data.get("confidence", "medium")).lower()
            if confidence not in ("low", "medium", "high"):
                confidence = "medium"

            parsed = {
                "chain_of_thought": str(data.get("chain_of_thought", "")),
                "where_it_went_wrong": str(data.get("where_it_went_wrong", "")),
                "root_cause": str(data.get("root_cause", "")),
                "suggested_fix": str(data.get("suggested_fix", "")),
                "confidence": confidence,
            }
            return parsed, False
    except Exception:
        pass

    # Fallback for invalid JSON: preserve raw text
    fallback_root = content.strip().split("\n")[0][:200] if content.strip() else "Unknown failure"
    return {
        "chain_of_thought": content,
        "where_it_went_wrong": content,
        "root_cause": fallback_root,
        "suggested_fix": "Review failure logs and diff.",
        "confidence": "low",
    }, True


class QwenClient:
    """Client for Qwen reasoning via llama-server."""

    def __init__(self, config: Optional[QwenConfig] = None):
        self.config = config or QwenConfig()
        self.enabled = self.config.enabled
        self.url = self.config.url.rstrip("/")
        self.model = self.config.model
        self.timeout = self.config.timeout
        self.max_diff_chars = self.config.max_diff_chars
        self.feedback = self.config.feedback
        self._health_checked = False

    def check_health(self) -> bool:
        """Health check once at run start (only if enabled).

        If unreachable: warn once, disable for the run, continue.
        Also resolves model name from server if not explicitly configured.
        """
        if not self.enabled:
            return False
        if self._health_checked:
            return self.enabled

        self._health_checked = True
        endpoint = f"{self.url}/v1/models"
        try:
            with httpx.Client(timeout=min(self.timeout, 5.0)) as client:
                resp = client.get(endpoint)
                if resp.status_code == 200:
                    data = resp.json()
                    if not self.model and isinstance(data, dict) and "data" in data and len(data["data"]) > 0:
                        first_model = data["data"][0]
                        if isinstance(first_model, dict) and "id" in first_model:
                            self.model = first_model["id"]
                    if not self.model:
                        self.model = "default"
                    return True
                else:
                    logger.warning(
                        f"[QWEN] llama-server at {self.url} returned HTTP {resp.status_code}. "
                        "Disabling Qwen post-mortem for this run."
                    )
                    self.enabled = False
                    return False
        except Exception as exc:
            logger.warning(
                f"[QWEN] llama-server at {self.url} unreachable: {exc}. "
                "Disabling Qwen post-mortem for this run."
            )
            self.enabled = False
            return False

    def run_postmortem(
        self,
        ticket: str,
        rejecting_source: str,
        rejection_reason: str,
        jev_verdict: Optional[Any],
        diff: str,
        strike_number: str,
        previous_strikes: Optional[List[Dict[str, Any]]] = None,
        thread_id: Optional[str] = None,
        subgoal_id: Optional[str] = None,
        strike_idx: int = 1,
    ) -> Optional[QwenPostmortemResult]:
        """Runs the advisory post-mortem via llama-server.

        Never alters the gate or raises exceptions into the engine.
        """
        if not self.enabled:
            return None

        prompt = format_postmortem_prompt(
            ticket=ticket,
            rejecting_source=rejecting_source,
            rejection_reason=rejection_reason,
            jev_verdict=jev_verdict,
            diff=diff,
            strike_number=strike_number,
            previous_strikes=previous_strikes,
            max_diff_chars=self.max_diff_chars,
        )

        model_to_use = self.model or "default"
        payload = {
            "model": model_to_use,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert software engineer analyzing code rejections and providing advisory post-mortems.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }

        endpoint = f"{self.url}/v1/chat/completions"
        start_time = time.perf_counter()

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(endpoint, json=payload)
                latency_ms = (time.perf_counter() - start_time) * 1000

                if resp.status_code != 200:
                    err_msg = f"HTTP {resp.status_code}: {resp.text[:300]}"
                    result = QwenPostmortemResult(
                        chain_of_thought="",
                        where_it_went_wrong="",
                        root_cause="",
                        suggested_fix="",
                        confidence="low",
                        raw_text=resp.text,
                        latency_ms=latency_ms,
                        error=err_msg,
                        parse_fallback=True,
                    )
                    self._persist_markdown_safe(
                        thread_id=thread_id,
                        strike_idx=strike_idx,
                        subgoal_id=subgoal_id,
                        strike_number=strike_number,
                        rejecting_source=rejecting_source,
                        result=result,
                    )
                    return result

                data = resp.json()
                content = ""
                if isinstance(data, dict) and "choices" in data and len(data["choices"]) > 0:
                    choice = data["choices"][0]
                    if isinstance(choice, dict) and "message" in choice:
                        content = choice["message"].get("content", "")

                parsed_dict, is_fallback = parse_qwen_response(content)

                result = QwenPostmortemResult(
                    chain_of_thought=parsed_dict["chain_of_thought"],
                    where_it_went_wrong=parsed_dict["where_it_went_wrong"],
                    root_cause=parsed_dict["root_cause"],
                    suggested_fix=parsed_dict["suggested_fix"],
                    confidence=parsed_dict["confidence"],
                    raw_text=content if is_fallback else None,
                    latency_ms=latency_ms,
                    error=None,
                    parse_fallback=is_fallback,
                )

                self._persist_markdown_safe(
                    thread_id=thread_id,
                    strike_idx=strike_idx,
                    subgoal_id=subgoal_id,
                    strike_number=strike_number,
                    rejecting_source=rejecting_source,
                    result=result,
                )
                return result

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            err_msg = f"{type(exc).__name__}: {exc}"
            result = QwenPostmortemResult(
                chain_of_thought="",
                where_it_went_wrong="",
                root_cause="",
                suggested_fix="",
                confidence="low",
                raw_text=None,
                latency_ms=latency_ms,
                error=err_msg,
                parse_fallback=True,
            )
            self._persist_markdown_safe(
                thread_id=thread_id,
                strike_idx=strike_idx,
                subgoal_id=subgoal_id,
                strike_number=strike_number,
                rejecting_source=rejecting_source,
                result=result,
            )
            return result

    def _persist_markdown_safe(
        self,
        thread_id: Optional[str],
        strike_idx: int,
        subgoal_id: Optional[str],
        strike_number: str,
        rejecting_source: str,
        result: QwenPostmortemResult,
    ) -> None:
        """Writes markdown copy under ~/.jev/postmortems/<thread_id>/strike-<n>.md."""
        if not thread_id:
            return
        try:
            postmortems_dir = get_jev_home() / "postmortems" / thread_id
            postmortems_dir.mkdir(parents=True, exist_ok=True)
            target_file = postmortems_dir / f"strike-{strike_idx}.md"

            md_content = (
                f"# Qwen Advisory Post-Mortem: Strike {strike_idx}\n\n"
                f"- **Thread ID**: {thread_id}\n"
                f"- **Subgoal ID**: {subgoal_id or 'unknown'}\n"
                f"- **Strike**: {strike_number}\n"
                f"- **Rejecting Source**: {rejecting_source}\n"
                f"- **Confidence**: {result.confidence}\n"
                f"- **Latency**: {result.latency_ms:.1f} ms\n"
                f"- **Error**: {result.error or 'None'}\n\n"
                f"## Root Cause\n{result.root_cause or '(None)'}\n\n"
                f"## Where It Went Wrong\n{result.where_it_went_wrong or '(None)'}\n\n"
                f"## Suggested Fix\n{result.suggested_fix or '(None)'}\n\n"
                f"## Chain of Thought\n{result.chain_of_thought or '(None)'}\n"
            )
            if result.raw_text:
                md_content += f"\n## Raw Output\n```\n{result.raw_text}\n```\n"

            target_file.write_text(md_content, encoding="utf-8")
        except Exception as exc:
            logger.warning(f"[QWEN] Could not write post-mortem markdown: {exc}")

    def format_advisory_feedback(self, result: Optional[QwenPostmortemResult]) -> str:
        """Formats advisory feedback block for retry prompt when feedback is enabled."""
        if not result or result.error or not self.feedback:
            return ""

        where = result.where_it_went_wrong.strip() if result.where_it_went_wrong else ""
        fix = result.suggested_fix.strip() if result.suggested_fix else ""
        if not where and not fix:
            return ""

        return (
            "\n\n[Advisory analysis, may be wrong]\n"
            f"Where it went wrong: {where}\n"
            f"Suggested fix: {fix}"
        )
