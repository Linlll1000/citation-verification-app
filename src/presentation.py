"""Presentation helpers shared by the Streamlit table and CSV export."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

from src.schemas import AnalysisRow, EvidenceBasis, MethodResult, SourceStatus, SupportBand


OUTPUT_COLUMNS = [
    "pair_id",
    "引用陳述",
    "引用標記",
    "ref 編號",
    "被引文獻",
    "ref 配對狀態",
    "錯誤狀態",
    "來源狀態",
    "方法 A 摘要來源",
    "方法 A 支持分數",
    "方法 A 證據層級",
    "方法 A 判斷依據",
    "方法 A 篇名判斷",
    "方法 A 判斷狀態",
    "摘要查詢紀錄",
    "摘要來源網址",
    "摘要 HTTP 請求紀錄",
    "方法 B 支持分數",
    "方法 B 證據層級",
    "方法 B 判斷依據",
    "方法 A 查核紀錄",
    "共引群組",
    "查核子主張",
    "共引摘要涵蓋",
    "共引待查數",
    "方法 B 判斷狀態",
    "方法 B 查核紀錄",
    "方法 A 證據解讀",
    "方法 A 摘要節錄",
]

SUPPORT_BAND_COLORS = {
    SupportBand.SUPPORTED.value: ("#E8F5E9", "#1B5E20"),
    SupportBand.UNCERTAIN.value: ("#FFF3E0", "#A65300"),
    SupportBand.UNSUPPORTED.value: ("#FDECEC", "#B42318"),
    SupportBand.UNVERIFIABLE.value: ("#F2F4F7", "#475467"),
}


@dataclass(frozen=True, slots=True)
class SharedClaimCoverage:
    group_id: str
    original_claim: str
    total: int
    supported: int
    pending: int


def summarize_shared_claims(rows: list[AnalysisRow]) -> list[SharedClaimCoverage]:
    """Count each subclaim once, using abstract judgments for coverage."""
    groups: dict[tuple[str, str], dict[str, list[MethodResult | None]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        pair = row.pair
        if pair.claim_group_id:
            key = (pair.claim_group_id, pair.original_claim_text)
            groups[key][pair.claim_text].append(row.method_a)

    coverage = []
    for (group_id, original_claim), subclaims in groups.items():
        supported = pending = 0
        for results in subclaims.values():
            if any(_abstract_supports(result) for result in results):
                supported += 1
            elif any(_abstract_check_pending(result) for result in results):
                pending += 1
        coverage.append(SharedClaimCoverage(group_id, original_claim, len(subclaims), supported, pending))
    return coverage


def _abstract_supports(result: MethodResult | None) -> bool:
    return (
        result is not None
        and result.source_status is SourceStatus.GOT_ABSTRACT
        and result.evidence_basis is EvidenceBasis.ABSTRACT
        and result.band is SupportBand.SUPPORTED
    )


def _abstract_check_pending(result: MethodResult | None) -> bool:
    if result is None or result.band is None:
        return True
    if result.source_status is SourceStatus.SKIPPED:
        return False
    return result.evidence_basis is not EvidenceBasis.ABSTRACT or result.source_status is not SourceStatus.GOT_ABSTRACT


def build_output_dataframe(rows: list[AnalysisRow]) -> pd.DataFrame:
    """Build the stable one-pair-per-row website and CSV table."""
    coverage_by_id = {group.group_id: group for group in summarize_shared_claims(rows)}
    records = []
    for row in rows:
        record = row.to_record()
        group = coverage_by_id.get(row.pair.claim_group_id)
        if group:
            record["共引摘要涵蓋"] = f"{group.supported}/{group.total} 項"
            record["共引待查數"] = str(group.pending)
        else:
            record["共引摘要涵蓋"] = ""
            record["共引待查數"] = ""
        records.append(record)
    return pd.DataFrame(records, columns=OUTPUT_COLUMNS)


def support_band_style(value: object) -> str:
    """Return accessible background/text colors for a temporary band label."""

    colors = SUPPORT_BAND_COLORS.get(str(value))
    if colors is None:
        return ""
    background, foreground = colors
    return f"background-color: {background}; color: {foreground}; font-weight: 600"


def style_output_dataframe(frame: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Color only the two support-band columns, leaving evidence data intact."""

    band_columns = ["方法 A 證據層級", "方法 B 證據層級"]
    return frame.style.map(support_band_style, subset=band_columns)


def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Export Excel-friendly UTF-8 CSV with a byte-order mark."""

    return frame.to_csv(index=False).encode("utf-8-sig")

