"""Stable interfaces for integrating the second and third team members."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from src.schemas import CitationPair, EvidenceBasis, SourceStatus


class TitleRelevance(StrEnum):
    """Temporary three-choice output of the Jev title question."""

    RELATED = "相關"
    UNCERTAIN = "不確定"
    UNRELATED = "不相關"


class JudgmentStatus(StrEnum):
    COMPLETED = "完成"
    STOPPED = "Q1 停止"
    MISSING_TITLE = "缺少篇名"
    REFERENCE_ERROR = "書目處理失敗"
    TITLE_ERROR = "Q1 失敗"
    SUPPORT_ERROR = "Q2 失敗"
    BATCH_STOPPED = "API 暫停"
    EVIDENCE_UNAVAILABLE = "查詢失敗，暫不評分"
    TITLE_AFTER_LOOKUP_FAILURE = "摘要查詢失敗，依篇名評分"
    UNVERIFIABLE = "無內容證據，無法查核"
    MOCK = "模擬結果"


@dataclass(frozen=True, slots=True)
class ReferenceLookupResult:
    """Data contract supplied by the second team member."""

    ref_number: int
    title: str
    doi: str = ""
    url: str = ""
    abstract: str = ""
    source_status: SourceStatus = SourceStatus.PENDING
    error_message: str = ""
    source_note: str = ""
    abstract_source: str = ""
    abstract_url: str = ""
    attempts: tuple[dict, ...] = ()


@dataclass(frozen=True, slots=True)
class JevJudgmentResult:
    """Data contract supplied by the third team member."""

    title_choice: TitleRelevance | None = None
    title_probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None
    support_score: float | None = None
    evidence_basis: EvidenceBasis | None = None
    status: JudgmentStatus = JudgmentStatus.COMPLETED
    error_message: str = ""
    raw_support_score: float | None = None
    title_model: str = ""
    title_request_id: str = ""
    support_model: str = ""
    support_request_id: str = ""
    stop_batch: bool = False
    evidence_excerpt: str = ""
    evidence_excerpt_status: str = ""
    evidence_excerpt_request_id: str = ""
    support_source_title: str = ""

    def __post_init__(self) -> None:
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if self.support_score is not None and not 0.1 <= self.support_score <= 0.99:
            raise ValueError("support_score must be between 0.1 and 0.99")
        if self.title_probabilities:
            total = sum(self.title_probabilities.values())
            if any(not 0.0 <= value <= 1.0 for value in self.title_probabilities.values()):
                raise ValueError("title probabilities must be between 0.0 and 1.0")
            if abs(total - 1.0) > 0.01 + 1e-12:
                raise ValueError("title probabilities must sum to 1.0")


class ReferenceLookupAdapter(Protocol):
    """Interface implemented by the second team member's lookup module."""

    def lookup(self, pair: CitationPair) -> ReferenceLookupResult:
        """Return metadata, abstract and source status after Q1 passes.

        Error messages returned in the result must be safe for the table and CSV.
        """


class JevAdapter(Protocol):
    """Interface implemented by the third team member's Jev module."""

    def judge(
        self,
        pair: CitationPair,
        reference: ReferenceLookupResult,
        *,
        lookup_abstract: Callable[[], ReferenceLookupResult] | None = None,
    ) -> JevJudgmentResult:
        """Judge one pair, calling the optional lookup only after Q1 passes."""

