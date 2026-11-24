"""Judge supplied citation text and bibliography with a general-purpose LLM."""

from collections.abc import Callable
from dataclasses import replace
from http.client import HTTPException
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src.adapters import ReferenceLookupResult
from src.judging_standard import SUPPORT_STANDARD
from src.schemas import (
    AnalysisRow,
    CitationPair,
    DirectJudgment,
    EvidenceBasis,
    MethodResult,
    ReferenceMatchStatus,
    SourceStatus,
    SupportBand,
)


CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_METHOD_B_MODEL = "openai/gpt-4.1-mini"

_INSTRUCTIONS = (
    "Judge each listed citation pair independently. " + SUPPORT_STANDARD + " "
    "Each pair carries its own `evidence` (text from the cited source); use only that pair's evidence. "
    "The paragraph supplies context for interpreting the claim; it is not evidence for itself. "
    "Do not use tools, search, memory, or evidence from other pairs. "
    "Return exactly one result per pair_id. Give a score from 0.10 to 0.99 for "
    "how strongly the evidence supports what the sentence attributes to that source. "
    "Scores of at least 0.70 mean supported, 0.40 to less than 0.70 mean uncertain, "
    "and below 0.40 mean unsupported."
)
UNVERIFIABLE_STATUS = "無內容證據，無法查核"


_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "citation_judgments",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pair_id": {"type": "string"},
                            "support_score": {"type": "number"},
                        },
                        "required": ["pair_id", "support_score"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["results"],
            "additionalProperties": False,
        },
    },
}


class MethodBError(Exception):
    """A response or request failure safe to show to the user."""


def run_method_b(
    rows: list[AnalysisRow],
    paragraph: str,
    bibliography: str,
    api_key: str,
    *,
    model: str = DEFAULT_METHOD_B_MODEL,
    evidence_lookup: Callable[[CitationPair], ReferenceLookupResult] | None = None,
) -> list[AnalysisRow]:
    """Judge with the same evidence method A uses (the cited source's abstract).

    ``bibliography`` is kept for compatibility but no longer sent: the reference
    string is not evidence. Pairs method A never looked up (title stage stopped)
    are looked up with ``evidence_lookup`` so both methods see the same pairs.
    """
    evidence: dict[str, str] = {}
    for row in rows:
        if row.pair.match_status is not ReferenceMatchStatus.MATCHED:
            continue
        reference = row.reference
        if (reference is None or reference.source_status in (SourceStatus.PENDING, SourceStatus.SKIPPED)) \
                and evidence_lookup is not None:
            try:
                reference = evidence_lookup(row.pair)
            except Exception:
                reference = None
        if reference is not None and reference.source_status is SourceStatus.GOT_ABSTRACT \
                and reference.abstract.strip():
            evidence[row.pair.pair_id] = reference.abstract.strip()

    targets = {
        row.pair.pair_id: {"pair_id": row.pair.pair_id, "claim": row.pair.claim_text,
                           "cited_as": row.pair.citation_marker,
                           "evidence": evidence[row.pair.pair_id]}
        for row in rows
        if row.pair.pair_id in evidence
    }
    results = {}
    duplicates: set[str] = set()
    reported_model = request_id = error = ""
    if targets:
        try:
            answers, reported_model, request_id = _request_results(
                paragraph, list(targets.values()), api_key, model
            )
        except MethodBError as exc:
            error = str(exc)
        else:
            for answer in answers:
                pair_id = answer.get("pair_id") if isinstance(answer, dict) else None
                if isinstance(pair_id, str) and pair_id in targets:
                    if pair_id in results:
                        duplicates.add(pair_id)
                    results[pair_id] = answer

    updated = []
    for row in rows:
        pair_id = row.pair.pair_id
        if row.pair.match_status is not ReferenceMatchStatus.MATCHED:
            status = "缺少 ref，暫不評分"
        elif pair_id not in targets:
            updated.append(replace(
                row,
                method_b=MethodResult(None, EvidenceBasis.UNAVAILABLE,
                                      band_override=SupportBand.UNVERIFIABLE),
                direct_judgment=DirectJudgment(UNVERIFIABLE_STATUS, reported_model, request_id),
            ))
            continue
        elif error:
            status = error
        elif pair_id in duplicates:
            status = "回覆包含重複配對，暫不評分"
        elif pair_id not in results:
            status = "回覆缺少此配對，暫不評分"
        else:
            updated.append(_apply_result(row, results[pair_id], reported_model, request_id))
            continue
        updated.append(replace(row, method_b=None, direct_judgment=DirectJudgment(status, reported_model, request_id)))
    return updated


def _request_results(
    paragraph: str, targets: list[dict], api_key: str, model: str,
) -> tuple[list, str, str]:
    if model != DEFAULT_METHOD_B_MODEL:
        raise MethodBError("方法 B 目前只使用已核對的 GPT-4.1 Mini，暫不評分")
    body = {
        "model": model,
        "temperature": 0,
        "provider": {"require_parameters": True},
        "plugins": [{"id": "web", "enabled": False}],
        "messages": [
            {"role": "system", "content": _INSTRUCTIONS},
            {"role": "user", "content": json.dumps({
                "paragraph": paragraph, "citation_pairs": targets,
            }, ensure_ascii=False)},
        ],
        "response_format": _RESPONSE_FORMAT,
    }
    request = Request(CHAT_URL, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
    request.add_unredirected_header("Authorization", f"Bearer {api_key}")
    try:
        with urlopen(request, timeout=40) as response:
            payload = json.load(response)
        choice = payload["choices"][0]
        if choice.get("finish_reason") in ("length", "content_filter", "error"):
            raise MethodBError("模型回覆未完成，暫不評分")
        message = choice["message"]
        tool_use = (payload.get("usage") or {}).get("server_tool_use") or {}
        if (message.get("annotations") or message.get("tool_calls")
                or tool_use.get("web_search_requests") or tool_use.get("web_fetch_requests")):
            raise MethodBError("OpenRouter 回報外部工具使用，方法 B 暫不評分")
        results = json.loads(message["content"])["results"]
        if not isinstance(results, list):
            raise TypeError("Expected a list of citation decisions")
    except HTTPError as exc:
        status = exc.code
        exc.close()
        raise MethodBError(f"OpenRouter 請求失敗（HTTP {status}）") from None
    except (OSError, HTTPException):
        raise MethodBError("無法連線至 OpenRouter，或請求已逾時") from None
    except (AttributeError, KeyError, IndexError, TypeError, ValueError):
        raise MethodBError("模型回覆格式有誤，暫不評分") from None

    return results, payload.get("model") or model, payload.get("id") or ""


def _apply_result(row: AnalysisRow, answer: dict, model: str, request_id: str) -> AnalysisRow:
    raw_score = answer.get("support_score")
    if type(raw_score) not in (int, float) or not 0 <= raw_score <= 1:
        return replace(row, method_b=None,
                       direct_judgment=DirectJudgment("此配對的回覆格式有誤，暫不評分", model, request_id))

    score = min(0.99, max(0.10, float(raw_score)))
    return replace(
        row,
        method_b=MethodResult(score, EvidenceBasis.ABSTRACT),
        direct_judgment=DirectJudgment("完成", model, request_id, float(raw_score)),
    )
