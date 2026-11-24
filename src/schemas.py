"""Shared data types for the citation-verification prototype.

The support bands are temporary engineering labels from the current meeting
decision. They are kept separate from source availability and evidence level
so later research revisions do not require rewriting the website.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.adapters import JevJudgmentResult, ReferenceLookupResult


_ABSTRACT_SOURCE_NAMES = {"semantic_scholar": "Semantic Scholar", "openalex": "OpenAlex"}


class SourceStatus(StrEnum):
    """Whether a cited source can currently be verified."""

    GOT_ABSTRACT = "拿到摘要"
    NO_ABSTRACT = "有文獻但沒摘要"
    NOT_FOUND = "查不到"
    PENDING = "待查詢"
    SKIPPED = "Q1 停止未查詢"
    LOOKUP_FAILED = "查詢失敗"


class EvidenceBasis(StrEnum):
    """The strongest evidence actually used for a judgment."""

    TITLE = "篇名"
    ABSTRACT = "摘要"
    FULL_TEXT = "全文"
    METADATA = "書目資料"
    UNAVAILABLE = "無可用證據"


class SupportBand(StrEnum):
    """Temporary three-level display label used by the prototype."""

    SUPPORTED = "支持"
    UNCERTAIN = "不確定"
    UNSUPPORTED = "不支持"
    UNVERIFIABLE = "無法查核"


class ReferenceMatchStatus(StrEnum):
    """Whether an in-text citation number has a pasted ref entry."""

    MATCHED = "已配對"
    MISSING_REFERENCE = "缺少 ref"


def support_band(score: float) -> SupportBand:
    """Map a 0.1-0.99 score to the temporary display band."""

    if not 0.1 <= score <= 0.99:
        raise ValueError("support_score must be between 0.1 and 0.99")
    if score >= 0.70:
        return SupportBand.SUPPORTED
    if score >= 0.40:
        return SupportBand.UNCERTAIN
    return SupportBand.UNSUPPORTED


@dataclass(frozen=True, slots=True)
class CitationPair:
    """One citation claim paired with one referenced source."""

    pair_id: str
    claim_text: str
    citation_marker: str
    ref_number: int
    ref_raw: str
    is_compound: bool = False
    match_status: ReferenceMatchStatus = ReferenceMatchStatus.MATCHED
    error_message: str = ""
    original_claim_text: str = ""
    claim_group_id: str = ""


@dataclass(frozen=True, slots=True)
class MethodResult:
    """A method's result for one claim-source pair."""

    support_score: float | None
    evidence_basis: EvidenceBasis | None
    source_status: SourceStatus = SourceStatus.PENDING
    band_override: SupportBand | None = None

    def __post_init__(self) -> None:
        if self.support_score is not None and not 0.1 <= self.support_score <= 0.99:
            raise ValueError("support_score must be between 0.1 and 0.99")

    @property
    def band(self) -> SupportBand | None:
        if self.band_override is not None:
            return self.band_override
        if self.support_score is None:
            return None
        return support_band(self.support_score)


@dataclass(frozen=True, slots=True)
class DirectJudgment:
    """Method B's request and evidence record for one pair."""

    status: str
    model: str = ""
    request_id: str = ""
    raw_score: float | None = None


@dataclass(frozen=True, slots=True)
class AnalysisRow:
    """One output-table row shared by the website and CSV exporter."""

    pair: CitationPair
    method_a: MethodResult | None = None
    method_b: MethodResult | None = None
    judgment: JevJudgmentResult | None = None
    reference: ReferenceLookupResult | None = None
    direct_judgment: DirectJudgment | None = None

    def to_record(self) -> dict[str, object]:
        """Return a flat, stable record for Streamlit and CSV output."""

        judgment = self.judgment
        direct = self.direct_judgment
        title_choice = judgment.title_choice if judgment else None
        abstract_source = ""
        if self.reference and self.method_a and self.method_a.evidence_basis is EvidenceBasis.ABSTRACT:
            abstract_source = self.reference.abstract_source
        errors = (
            self.pair.error_message,
            self.reference.error_message if self.reference else "",
            judgment.error_message if judgment else "",
        )
        return {
            "pair_id": self.pair.pair_id,
            "引用陳述": self.pair.original_claim_text or self.pair.claim_text,
            "引用標記": self.pair.citation_marker,
            "ref 編號": self.pair.ref_number,
            "被引文獻": self.pair.ref_raw,
            "ref 配對狀態": self.pair.match_status.value,
            "錯誤狀態": "；".join(filter(None, errors)),
            "來源狀態": self.method_a.source_status.value if self.method_a else "",
            "方法 A 摘要來源": _ABSTRACT_SOURCE_NAMES.get(abstract_source, abstract_source),
            "方法 A 支持分數": _score(self.method_a),
            "方法 A 證據層級": _band(self.method_a),
            "方法 A 判斷依據": _evidence(self.method_a),
            "方法 A 篇名判斷": title_choice.value if title_choice else "",
            "方法 A 判斷狀態": judgment.status.value if judgment else "",
            "方法 A 證據解讀": self._method_a_evidence_scope(),
            "方法 A 摘要節錄": (judgment.evidence_excerpt or judgment.evidence_excerpt_status) if judgment else "",
            "摘要查詢紀錄": self.reference.source_note if self.reference else "",
            "摘要來源網址": self.reference.abstract_url if self.reference else "",
            "摘要 HTTP 請求紀錄": json.dumps(self.reference.attempts, ensure_ascii=False)
            if self.reference and self.reference.attempts else "",
            "方法 B 支持分數": _score(self.method_b),
            "方法 B 證據層級": _band(self.method_b),
            "方法 B 判斷依據": _evidence(self.method_b),
            "方法 B 判斷狀態": direct.status if direct else "",
            "方法 B 查核紀錄": json.dumps(
                {"model": direct.model, "request_id": direct.request_id, "raw_score": direct.raw_score},
                ensure_ascii=False,
            ) if direct else "",
            "方法 A 查核紀錄": self._method_a_evidence_record(),
            "共引群組": self.pair.claim_group_id,
            "查核子主張": (
                self.pair.claim_text if self.pair.original_claim_text or self.pair.claim_group_id else ""
            ),
        }

    def _method_a_evidence_record(self) -> str:
        judgment, reference = self.judgment, self.reference
        if judgment is None or judgment.raw_support_score is None or reference is None:
            return ""
        abstract_used = judgment.evidence_basis is EvidenceBasis.ABSTRACT
        record = {
            "abstract_source": reference.abstract_source if abstract_used else "",
            "abstract_url": reference.abstract_url if abstract_used else "",
            "evidence": (reference.abstract if abstract_used else reference.title).strip(),
            "model": judgment.support_model,
            "request_id": judgment.support_request_id,
            "raw_score": judgment.raw_support_score,
        }
        if judgment.support_source_title:
            record["verified_source_title"] = judgment.support_source_title
        if judgment.evidence_excerpt_status:
            record["excerpt_check"] = {
                "status": judgment.evidence_excerpt_status,
                "excerpt": judgment.evidence_excerpt,
                "request_id": judgment.evidence_excerpt_request_id,
            }
        return json.dumps(record, ensure_ascii=False)

    def _method_a_evidence_scope(self) -> str:
        result = self.method_a
        if result is None or result.band is None:
            return ""
        if result.source_status is SourceStatus.SKIPPED:
            return "篇名不相關，未進行 Q2 證據判斷"
        if result.evidence_basis is EvidenceBasis.TITLE:
            return (
                "證據不足，只有篇名可查核" if result.band is SupportBand.UNSUPPORTED
                else "僅依篇名，尚未查核摘要"
            )
        if result.evidence_basis is EvidenceBasis.ABSTRACT and result.band is SupportBand.UNSUPPORTED:
            if self.pair.is_compound and not self.pair.original_claim_text:
                return "這一篇摘要未能支持整句，未逐項判斷"
            return "摘要未能支持這項主張，未判定有反證"
        return ""


def _score(result: MethodResult | None) -> float | None:
    return None if result is None else result.support_score


def _evidence(result: MethodResult | None) -> str:
    if result is None or result.evidence_basis is None:
        return ""
    return result.evidence_basis.value


def _band(result: MethodResult | None) -> str:
    if result is None or result.band is None:
        return ""
    return result.band.value

