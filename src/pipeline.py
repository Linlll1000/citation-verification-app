"""Orchestrate adapters without coupling the website to either implementation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from src.adapters import (
    JevAdapter,
    JevJudgmentResult,
    JudgmentStatus,
    ReferenceLookupAdapter,
    ReferenceLookupResult,
)
from src.reference_metadata import extract_reference_metadata
from src.schemas import (
    AnalysisRow,
    CitationPair,
    MethodResult,
    ReferenceMatchStatus,
    SourceStatus,
    SupportBand,
)


def run_method_a_pipeline(
    pairs: list[CitationPair],
    reference_adapter: ReferenceLookupAdapter | None,
    jev_adapter: JevAdapter,
    *,
    prepare_reference: Callable[[CitationPair], ReferenceLookupResult] | None = None,
) -> list[AnalysisRow]:
    """Read local metadata first and defer every abstract lookup until Q1 passes."""

    rows: list[AnalysisRow] = []
    batch_error = ""
    for pair in pairs:
        if pair.match_status is ReferenceMatchStatus.MISSING_REFERENCE:
            rows.append(AnalysisRow(pair=pair))
            continue

        if batch_error:
            reference = ReferenceLookupResult(pair.ref_number, "")
            judgment = JevJudgmentResult(status=JudgmentStatus.BATCH_STOPPED, error_message=batch_error)
        else:
            reference, judgment = _judge_pair(pair, reference_adapter, jev_adapter, prepare_reference)
            if judgment.stop_batch:
                batch_error = judgment.error_message
        source_status = reference.source_status
        stopped = judgment.status is JudgmentStatus.STOPPED
        if stopped and source_status is SourceStatus.PENDING:
            source_status = SourceStatus.SKIPPED
        method_a = MethodResult(
            support_score=judgment.support_score,
            evidence_basis=judgment.evidence_basis,
            source_status=source_status,
            band_override=(
                SupportBand.UNSUPPORTED if stopped
                else SupportBand.UNVERIFIABLE if judgment.status is JudgmentStatus.UNVERIFIABLE
                else None
            ),
        )
        rows.append(
            AnalysisRow(
                pair=pair,
                method_a=method_a,
                judgment=judgment,
                reference=reference,
            )
        )
    return rows


def _judge_pair(
    pair: CitationPair,
    reference_adapter: ReferenceLookupAdapter | None,
    jev_adapter: JevAdapter,
    prepare_reference: Callable[[CitationPair], ReferenceLookupResult] | None,
) -> tuple[ReferenceLookupResult, JevJudgmentResult]:
    try:
        if prepare_reference is None:
            metadata = extract_reference_metadata(pair.ref_raw)
            reference = ReferenceLookupResult(
                ref_number=pair.ref_number,
                title=metadata.title,
                doi=metadata.doi,
                url=metadata.url,
                source_note=metadata.parse_note,
            )
        else:
            reference = _validate_reference(pair, prepare_reference(pair))
    except Exception as error:
        return ReferenceLookupResult(pair.ref_number, ""), JevJudgmentResult(
            status=JudgmentStatus.REFERENCE_ERROR,
            error_message=f"無法讀取本筆書目資料（{type(error).__name__}）。",
        )
    if reference_adapter is None:
        return reference, jev_adapter.judge(pair, reference)

    def lookup_abstract() -> ReferenceLookupResult:
        nonlocal reference
        try:
            found = _validate_reference(pair, reference_adapter.lookup(pair))
            reference = replace(
                found,
                title=reference.title or found.title,
                doi=reference.doi or found.doi,
                url=reference.url or found.url,
            )
        except Exception as error:
            reference = replace(
                reference,
                abstract="",
                abstract_source="",
                source_status=SourceStatus.LOOKUP_FAILED,
                error_message=f"摘要查詢失敗，無法確認摘要（{type(error).__name__}）。",
            )
        if reference.source_status is SourceStatus.NOT_FOUND and not reference.error_message:
            reference = replace(
                reference, error_message="摘要查詢未找到可確認的文獻。",
            )
        return reference

    judgment = jev_adapter.judge(pair, reference, lookup_abstract=lookup_abstract)
    return reference, judgment


def _validate_reference(pair: CitationPair, reference: ReferenceLookupResult) -> ReferenceLookupResult:
    """Validate the handoff before using evidence from another module."""
    if not isinstance(reference, ReferenceLookupResult):
        raise TypeError("Expected ReferenceLookupResult")
    if type(reference.ref_number) is not int or reference.ref_number != pair.ref_number:
        raise ValueError("Reference number does not match the citation pair")
    if not isinstance(reference.source_status, SourceStatus) or any(
        not isinstance(value, str)
        for value in (
            reference.title,
            reference.doi,
            reference.url,
            reference.abstract,
            reference.error_message,
            reference.source_note,
            reference.abstract_source,
        )
    ):
        raise TypeError("Invalid reference fields")
    has_abstract = bool(reference.abstract.strip())
    if reference.source_status is SourceStatus.GOT_ABSTRACT and not has_abstract:
        raise ValueError("The source reported an abstract but supplied none")
    if reference.source_status is SourceStatus.NO_ABSTRACT and has_abstract:
        raise ValueError("The source reported no abstract but supplied one")
    return reference

